"""Worker 拥有的物理分页 K/V tensor。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from light_vllm.modeling.attention.interfaces import AttentionLayerSpec, ModelKVCacheSpec
from light_vllm.runtime.kv_cache import KVCacheError


@dataclass(frozen=True, slots=True)
class PagedKVCacheConfig:
    """所有 Worker 必须一致使用的物理页配置。"""

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


@dataclass(frozen=True, slots=True)
class PagedLayerCache:
    """一层采用 ``[block, offset, kv_head, head_size]`` 布局的 K/V。"""

    keys: Tensor
    values: Tensor


class PagedKVCache:
    """按全局物理 page ID 索引的逐层 K/V tensor 池。"""

    def __init__(self, model_spec: ModelKVCacheSpec, config: PagedKVCacheConfig) -> None:
        self._model_spec = model_spec
        self._config = config
        self._layer_specs = {layer.layer_id: layer for layer in model_spec.layers}
        self._layers = {layer.layer_id: self._allocate_layer(layer) for layer in model_spec.layers}

    @property
    def config(self) -> PagedKVCacheConfig:
        return self._config

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
    def write(self, layer_id: str, key: Tensor, value: Tensor, slot_mapping: Tensor) -> None:
        """把有效 query token 的 K/V 原位写到指定物理 slot。"""

        layer = self.layer(layer_id)
        spec = self.layer_spec(layer_id)
        expected_tail = (spec.num_kv_heads, spec.head_size)
        if key.ndim != 4 or key.shape[2:] != expected_tail:
            raise KVCacheError(
                "paged KV update must have shape [batch, query, kv_heads, head_size]"
            )
        if value.shape != key.shape:
            raise KVCacheError("paged K/V updates must have the same shape")
        if slot_mapping.shape != key.shape[:2]:
            raise KVCacheError("slot mapping must have one entry per query token")
        if key.dtype != self._config.dtype or key.device != self._config.device:
            raise KVCacheError("paged KV update dtype and device must match the cache")
        if value.dtype != key.dtype or value.device != key.device:
            raise KVCacheError("paged K/V updates must use the same dtype and device")
        if slot_mapping.device != self._config.device:
            raise KVCacheError("slot mapping device must match the cache")

        active = slot_mapping >= 0
        slots = slot_mapping[active].to(torch.long)
        if slots.numel() == 0:
            return
        num_slots = self._config.num_blocks * self._config.block_size
        if int(slots.min()) < 0 or int(slots.max()) >= num_slots:
            raise KVCacheError("slot mapping contains an out-of-range physical slot")
        if slots.unique().numel() != slots.numel():
            raise KVCacheError("slot mapping must not write the same physical slot twice")

        flat_keys = layer.keys.view(num_slots, *expected_tail)
        flat_values = layer.values.view(num_slots, *expected_tail)
        flat_keys.index_copy_(0, slots, key[active])
        flat_values.index_copy_(0, slots, value[active])

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
