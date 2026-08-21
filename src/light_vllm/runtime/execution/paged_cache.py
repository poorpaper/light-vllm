"""分页 Step Handler 拥有的物理 K/V tensor。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from light_vllm.modeling.attention.interfaces import AttentionLayerSpec, ModelKVCacheSpec
from light_vllm.runtime.kv_cache import KVCacheCapacityError, KVCacheError


class PagedKVCachePlanner(Protocol):
    """模型加载后，根据真实 KV 形状和设备内存决定分页缓存大小。"""

    @property
    def num_blocks(self) -> int: ...

    @property
    def block_size(self) -> int: ...

    @property
    def device(self) -> torch.device: ...

    def plan(self, model_spec: ModelKVCacheSpec) -> PagedKVCacheConfig: ...


@dataclass(frozen=True, slots=True)
class PagedKVCacheConfig:
    """已经确定的分页缓存配置；指定页数时也可以直接充当规划器。"""

    num_blocks: int
    block_size: int
    dtype: torch.dtype = torch.float32
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        if type(self.num_blocks) is not int or self.num_blocks <= 0:
            raise ValueError("num_blocks must be a positive integer")
        if type(self.block_size) is not int or self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if not self.dtype.is_floating_point:
            raise ValueError("KV cache dtype must be floating point")
        object.__setattr__(self, "device", torch.device(self.device))

    def plan(self, model_spec: ModelKVCacheSpec) -> PagedKVCacheConfig:
        return self


def kv_cache_bytes_per_block(
    model_spec: ModelKVCacheSpec,
    *,
    block_size: int,
    dtype: torch.dtype,
) -> int:
    """根据唯一的模型 KV 规格计算一个物理 block 的字节数。"""

    # 因子 2 分别代表 K 和 V；不同层可以声明不同的 KV head 形状。
    elements_per_token = sum(
        2 * layer.num_kv_heads * layer.head_size for layer in model_spec.layers
    )
    element_size = torch.empty((), dtype=dtype).element_size()
    return block_size * elements_per_token * element_size


@dataclass(slots=True)
class CudaMemoryKVCachePlanner:
    """模型加载后按当前空闲显存的一定比例规划物理页数。"""

    block_size: int
    dtype: torch.dtype
    device: str | torch.device
    memory_fraction: float = 0.8
    _config: PagedKVCacheConfig | None = None

    def __post_init__(self) -> None:
        if type(self.block_size) is not int or self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if not self.dtype.is_floating_point:
            raise ValueError("KV cache dtype must be floating point")
        self.device = torch.device(self.device)
        if self.device.type != "cuda":
            raise ValueError("CUDA memory discovery requires a CUDA device")
        if not 0 < self.memory_fraction <= 1:
            raise ValueError("memory_fraction must be within (0, 1]")

    @property
    def num_blocks(self) -> int:
        if self._config is None:
            raise KVCacheError("KV cache capacity is not available before planning")
        return self._config.num_blocks

    def plan(self, model_spec: ModelKVCacheSpec) -> PagedKVCacheConfig:
        # 模型权重加载到显卡后，剩余显存才是 KV cache 真正可以使用的空间。
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        budget_bytes = int(free_bytes * self.memory_fraction)
        bytes_per_block = kv_cache_bytes_per_block(
            model_spec,
            block_size=self.block_size,
            dtype=self.dtype,
        )
        # 只分配完整页，保证实际使用量不会超过显存预算。
        num_blocks = budget_bytes // bytes_per_block
        if num_blocks == 0:
            raise KVCacheCapacityError("available CUDA memory cannot hold one KV cache block")
        self._config = PagedKVCacheConfig(
            num_blocks=num_blocks,
            block_size=self.block_size,
            dtype=self.dtype,
            device=self.device,
        )
        return self._config


@dataclass(frozen=True, slots=True)
class PagedLayerCache:
    """一层采用 ``[block, offset, kv_head, head_size]`` 布局的 K/V。"""

    keys: Tensor
    values: Tensor


@dataclass(frozen=True, slots=True)
class PagedKVWriteMapping:
    """A slot mapping normalized once for every layer in one model step."""

    active: Tensor
    source_indices: Tensor
    slots: Tensor


class PagedKVCache:
    """所有请求共用的分页 K/V 张量池，通过 page ID 找到具体存储位置。"""

    def __init__(self, model_spec: ModelKVCacheSpec, config: PagedKVCacheConfig) -> None:
        self._model_spec = model_spec
        self._config = config
        self._layer_specs = {layer.layer_id: layer for layer in model_spec.layers}
        # 各层可能有不同 KV 形状，因此共享 page ID，但分别持有物理 tensor。
        self._layers = {layer.layer_id: self._allocate_layer(layer) for layer in model_spec.layers}

    @property
    def config(self) -> PagedKVCacheConfig:
        return self._config

    @property
    def model_spec(self) -> ModelKVCacheSpec:
        return self._model_spec

    def layer_spec(self, layer_id: str) -> AttentionLayerSpec:
        try:
            return self._layer_specs[layer_id]
        except KeyError as exc:
            raise KVCacheError(f"KV cache layer {layer_id!r} was not found") from exc

    def layer(self, layer_id: str) -> PagedLayerCache:
        try:
            return self._layers[layer_id]
        except KeyError as exc:
            raise KVCacheError(f"KV cache layer {layer_id!r} was not found") from exc

    @torch.inference_mode()
    def prepare_write(
        self,
        slot_mapping: Tensor,
        *,
        validate: bool = True,
    ) -> PagedKVWriteMapping:
        """Normalize one batch mapping for reuse across all model layers.

        Paged-attention metadata proves range and aliasing properties on the
        CPU. Its backends can therefore skip CUDA reductions and host
        synchronizations while direct cache callers keep defensive validation.
        """

        if slot_mapping.ndim != 1:
            raise KVCacheError("slot mapping must have one entry per packed query token")
        if slot_mapping.device != self._config.device:
            raise KVCacheError("slot mapping device must match the cache")
        active = slot_mapping >= 0
        source_indices = torch.nonzero(active.flatten(), as_tuple=False).flatten()
        slots = slot_mapping.flatten().index_select(0, source_indices).to(torch.long)
        if validate and slots.numel():
            num_slots = self._config.num_blocks * self._config.block_size
            if int(slots.min()) < 0 or int(slots.max()) >= num_slots:
                raise KVCacheError("slot mapping contains an out-of-range physical slot")
            if slots.unique().numel() != slots.numel():
                raise KVCacheError("slot mapping must not write the same physical slot twice")
        return PagedKVWriteMapping(
            active=active,
            source_indices=source_indices,
            slots=slots,
        )

    @torch.inference_mode()
    def write_prepared(
        self,
        layer_id: str,
        key: Tensor,
        value: Tensor,
        mapping: PagedKVWriteMapping,
    ) -> None:
        """Write K/V with a mapping prepared once for the model step."""

        layer = self.layer(layer_id)
        spec = self.layer_spec(layer_id)
        expected_tail = (spec.num_kv_heads, spec.head_size)
        if key.ndim != 3 or key.shape[1:] != expected_tail:
            raise KVCacheError("paged KV update must have shape [tokens, kv_heads, head_size]")
        if value.shape != key.shape:
            raise KVCacheError("paged K/V updates must have the same shape")
        if mapping.active.shape != key.shape[:1]:
            raise KVCacheError("slot mapping must have one entry per query token")
        if key.dtype != self._config.dtype or key.device != self._config.device:
            raise KVCacheError("paged KV update dtype and device must match the cache")
        if value.dtype != key.dtype or value.device != key.device:
            raise KVCacheError("paged K/V updates must use the same dtype and device")
        if (
            mapping.active.device != self._config.device
            or mapping.source_indices.device != self._config.device
            or mapping.slots.device != self._config.device
        ):
            raise KVCacheError("prepared slot mapping device must match the cache")
        if mapping.slots.numel() == 0:
            return

        num_slots = self._config.num_blocks * self._config.block_size
        flat_keys = layer.keys.view(num_slots, *expected_tail)
        flat_values = layer.values.view(num_slots, *expected_tail)
        query_keys = key
        query_values = value
        if mapping.source_indices.numel() != mapping.active.numel():
            query_keys = query_keys.index_select(0, mapping.source_indices)
            query_values = query_values.index_select(0, mapping.source_indices)
        flat_keys.index_copy_(0, mapping.slots, query_keys)
        flat_values.index_copy_(0, mapping.slots, query_values)

    @torch.inference_mode()
    def write(self, layer_id: str, key: Tensor, value: Tensor, slot_mapping: Tensor) -> None:
        """把有效 query token 的 K/V 原位写到指定物理 slot。"""

        self.write_prepared(layer_id, key, value, self.prepare_write(slot_mapping))

    @torch.inference_mode()
    def compact(
        self,
        block_ids: tuple[int, ...],
        *,
        num_computed_tokens: int,
        num_query_tokens: int,
        num_reserved_query_tokens: int,
        retained_query_indices: tuple[int, ...],
        num_readonly_prefix_blocks: int = 0,
    ) -> int:
        """把命中路径从展平 query slots 搬到连续正式尾部。"""

        block_ids = tuple(block_ids)
        retained = tuple(retained_query_indices)
        values = (
            num_computed_tokens,
            num_query_tokens,
            num_reserved_query_tokens,
            num_readonly_prefix_blocks,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("paged compact counts must be non-negative integers")
        if num_reserved_query_tokens < num_query_tokens:
            raise ValueError("reserved query tokens must cover every actual query")
        if any(type(index) is not int or index < 0 for index in retained):
            raise ValueError("retained query indices must be non-negative integers")
        if len(set(retained)) != len(retained):
            raise ValueError("retained query indices must be unique")
        if any(index >= num_query_tokens for index in retained):
            raise KVCacheError("retained query index exceeds the paged KV tail")

        config = self._config
        required_blocks = (
            num_computed_tokens + num_reserved_query_tokens + config.block_size - 1
        ) // config.block_size
        if len(block_ids) != required_blocks:
            raise KVCacheError("block table does not exactly cover the reserved query slots")
        if any(
            type(block_id) is not int or not 0 <= block_id < config.num_blocks
            for block_id in block_ids
        ):
            raise KVCacheError("block table contains an invalid physical block")
        if len(set(block_ids)) != len(block_ids):
            raise KVCacheError("block table aliases a physical block within one request")
        if num_readonly_prefix_blocks * config.block_size > num_computed_tokens:
            raise KVCacheError("readonly prefix blocks exceed the computed prefix")

        def physical_slot(logical_position: int) -> int:
            block_id = block_ids[logical_position // config.block_size]
            return block_id * config.block_size + logical_position % config.block_size

        source_slots = tuple(physical_slot(num_computed_tokens + index) for index in retained)
        target_slots = tuple(
            physical_slot(num_computed_tokens + index) for index in range(len(retained))
        )
        if source_slots == target_slots:
            return 0

        source = torch.tensor(source_slots, dtype=torch.long, device=config.device)
        target = torch.tensor(target_slots, dtype=torch.long, device=config.device)
        num_slots = config.num_blocks * config.block_size
        for layer in self._layers.values():
            flat_keys = layer.keys.view(num_slots, *layer.keys.shape[2:])
            flat_values = layer.values.view(num_slots, *layer.values.shape[2:])
            # 所有 source 先复制完成，再写 target，保证跨页重叠搬运正确。
            keys = flat_keys.index_select(0, source).clone()
            values = flat_values.index_select(0, source).clone()
            flat_keys.index_copy_(0, target, keys)
            flat_values.index_copy_(0, target, values)
        return sum(left != right for left, right in zip(source_slots, target_slots, strict=True))

    def _allocate_layer(self, spec: AttentionLayerSpec) -> PagedLayerCache:
        shape = (
            self._config.num_blocks,
            self._config.block_size,
            spec.num_kv_heads,
            spec.head_size,
        )
        return PagedLayerCache(
            keys=torch.empty(shape, dtype=self._config.dtype, device=self._config.device),
            values=torch.empty(shape, dtype=self._config.dtype, device=self._config.device),
        )
