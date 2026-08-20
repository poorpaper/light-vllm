"""可读性优先的 PyTorch Paged Attention 正确性实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from light_vllm.modeling.attention.interfaces import AttentionContext, AttentionLayerSpec
from light_vllm.runtime.execution.interfaces import QueryLayout
from light_vllm.runtime.execution.layout import query_visibility
from light_vllm.runtime.execution.paged_cache import PagedKVCache, PagedKVWriteMapping
from light_vllm.runtime.kv_cache import KVCacheError


@dataclass(frozen=True, slots=True)
class PagedAttentionMetadata:
    """一个 padded query batch 访问分页 K/V 所需的事实。"""

    block_tables: tuple[tuple[int, ...], ...]
    num_computed_tokens: tuple[int, ...]
    query_layouts: tuple[QueryLayout, ...]
    # 包含实际 query 和 Scheduler 预留但未使用的 query slots。
    num_reserved_query_tokens: tuple[int, ...] = ()
    num_readonly_prefix_blocks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        block_tables = tuple(tuple(table) for table in self.block_tables)
        computed = tuple(self.num_computed_tokens)
        layouts = tuple(self.query_layouts)
        reserved = tuple(self.num_reserved_query_tokens)
        readonly = tuple(self.num_readonly_prefix_blocks)
        num_requests = len(block_tables)
        if num_requests == 0:
            raise ValueError("paged attention metadata must contain at least one request")
        if len(computed) != num_requests or len(layouts) != num_requests:
            raise ValueError("paged attention metadata fields must have the same batch size")
        if not reserved:
            reserved = tuple(len(layout) for layout in layouts)
        if len(reserved) != num_requests:
            raise ValueError("reserved query counts must have the same batch size")
        if not readonly:
            readonly = (0,) * num_requests
        if len(readonly) != num_requests:
            raise ValueError("readonly prefix counts must have the same batch size")
        if any(type(value) is not int or value < 0 for value in computed):
            raise ValueError("computed-token counts must be non-negative integers")
        if any(
            type(value) is not int or value < len(layout)
            for value, layout in zip(reserved, layouts, strict=True)
        ):
            raise ValueError("reserved query counts must cover every actual query")
        if any(type(value) is not int or value < 0 for value in readonly):
            raise ValueError("readonly prefix counts must be non-negative integers")
        if any(
            type(block_id) is not int or block_id < 0
            for table in block_tables
            for block_id in table
        ):
            raise ValueError("block tables must contain non-negative integers")
        object.__setattr__(self, "block_tables", block_tables)
        object.__setattr__(self, "num_computed_tokens", computed)
        object.__setattr__(self, "query_layouts", layouts)
        object.__setattr__(self, "num_reserved_query_tokens", reserved)
        object.__setattr__(self, "num_readonly_prefix_blocks", readonly)

    @property
    def batch_size(self) -> int:
        return len(self.block_tables)

    @property
    def query_lengths(self) -> tuple[int, ...]:
        return tuple(len(layout) for layout in self.query_layouts)

    def slot_mapping(
        self,
        *,
        block_size: int,
        query_width: int,
        device: torch.device,
    ) -> Tensor:
        """把每个 query token 的展平位置转换成物理 cache slot。"""

        mapping: list[list[int]] = []
        for table, computed, layout, reserved in zip(
            self.block_tables,
            self.num_computed_tokens,
            self.query_layouts,
            self.num_reserved_query_tokens,
            strict=True,
        ):
            row = [-1] * query_width
            query_length = len(layout)
            if query_length > query_width:
                raise KVCacheError("query length exceeds the padded query width")
            required_blocks = (computed + reserved + block_size - 1) // block_size
            if len(table) < required_blocks:
                raise KVCacheError("block table does not cover all scheduled tokens")
            if len(table) > required_blocks:
                raise KVCacheError("block table contains unused physical blocks")
            for query_offset in range(query_length):
                logical_position = computed + query_offset
                block_id = table[logical_position // block_size]
                row[query_offset] = block_id * block_size + logical_position % block_size
            mapping.append(row)
        return torch.tensor(mapping, dtype=torch.long, device=device)

    def validate_block_tables(
        self,
        *,
        num_blocks: int,
        block_size: int,
    ) -> None:
        """校验本轮可能读取或写入的完整 block table。"""

        block_owners: dict[int, tuple[int, bool]] = {}
        for table, computed, reserved, num_readonly in zip(
            self.block_tables,
            self.num_computed_tokens,
            self.num_reserved_query_tokens,
            self.num_readonly_prefix_blocks,
            strict=True,
        ):
            required_blocks = (computed + reserved + block_size - 1) // block_size
            if len(table) < required_blocks:
                raise KVCacheError("block table does not cover all scheduled tokens")
            if len(table) > required_blocks:
                raise KVCacheError("block table contains unused physical blocks")
            if any(block_id >= num_blocks for block_id in table):
                raise KVCacheError("block table contains an out-of-range physical block")
            if num_readonly > len(table) or num_readonly * block_size > computed:
                raise KVCacheError("readonly prefix blocks exceed the computed prefix")
            if len(set(table)) != len(table):
                raise KVCacheError("block table aliases a physical block within one request")
            for logical_index, block_id in enumerate(table):
                readonly = logical_index < num_readonly
                previous = block_owners.get(block_id)
                if previous is not None and (
                    previous[0] != logical_index or not previous[1] or not readonly
                ):
                    raise KVCacheError("block tables alias a physical block across requests")
                block_owners[block_id] = (logical_index, readonly)

    def visibility_tensor(self, *, query_width: int, device: torch.device) -> Tensor:
        """构造 backend 内部使用的 padded query visibility。"""

        visibility: list[list[list[bool]]] = []
        for layout in self.query_layouts:
            length = len(layout)
            if length > query_width:
                raise KVCacheError("query layout exceeds the padded query width")
            visible_queries = query_visibility(layout)
            visibility.append(
                [list(row) + [False] * (query_width - length) for row in visible_queries]
                + [[False] * query_width for _ in range(query_width - length)]
            )
        return torch.tensor(visibility, dtype=torch.bool, device=device)

    def visible_logical_positions(
        self,
        row: int,
    ) -> tuple[tuple[int, ...], ...]:
        """一次推导某请求全部 query 可读取的逻辑位置。"""

        layout = self.query_layouts[row]
        computed = self.num_computed_tokens[row]
        prefix = tuple(range(computed))
        return tuple(
            prefix
            + tuple(computed + offset for offset, visible in enumerate(visible_queries) if visible)
            for visible_queries in query_visibility(layout)
        )


class PagedAttentionContext(AttentionContext, Protocol):
    """除模型需要的 attention 接口外，还记录本轮实际运行过哪些层。"""

    @property
    def layer_ids(self) -> frozenset[str]: ...


class PagedAttentionBackend(Protocol):
    """为本批请求创建分页 attention；PyTorch、CUDA 和 Triton 都实现它。"""

    def create(
        self,
        cache: PagedKVCache,
        metadata: PagedAttentionMetadata,
    ) -> PagedAttentionContext: ...


def _validate_paged_attention_tensors(
    cache: PagedKVCache,
    metadata: PagedAttentionMetadata,
    layer_id: str,
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> AttentionLayerSpec:
    """让不同 backend 共用同一套形状与页表校验。"""

    layer_spec = cache.layer_spec(layer_id)
    if query.ndim != 4:
        raise KVCacheError("paged attention query must have four dimensions")
    if query.shape[:2] != key.shape[:2] or value.shape != key.shape:
        raise KVCacheError("paged attention Q/K/V batch and query dimensions must match")
    if query.shape[0] != metadata.batch_size:
        raise KVCacheError("paged attention metadata batch size does not match query")
    if query.shape[2:] != (layer_spec.num_query_heads, layer_spec.head_size):
        raise KVCacheError("paged attention query shape does not match the layer spec")
    if key.shape[2:] != (layer_spec.num_kv_heads, layer_spec.head_size):
        raise KVCacheError("paged attention K/V shape does not match the layer spec")
    return layer_spec


class TorchPagedAttention:
    """逐物理页读取 K/V 的在线 softmax attention。

    这是便于阅读和测试的 PyTorch 参考实现。它不会先拼接完整历史 K/V，
    而是边读取每一页边累计 softmax。以后 CUDA/Triton 实现继续使用同一接口。
    """

    def __init__(self, cache: PagedKVCache, metadata: PagedAttentionMetadata) -> None:
        self._cache = cache
        self._metadata = metadata
        self._layer_ids: set[str] = set()
        self._write_mapping: PagedKVWriteMapping | None = None
        config = cache.config
        metadata.validate_block_tables(
            num_blocks=config.num_blocks,
            block_size=config.block_size,
        )

    @property
    def layer_ids(self) -> frozenset[str]:
        return frozenset(self._layer_ids)

    def forward(
        self,
        layer_id: str,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        scale: float,
    ) -> Tensor:
        if layer_id in self._layer_ids:
            raise KVCacheError(f"paged attention layer {layer_id!r} ran more than once")
        self._layer_ids.add(layer_id)
        _validate_paged_attention_tensors(
            self._cache,
            self._metadata,
            layer_id,
            query,
            key,
            value,
        )
        config = self._cache.config

        if self._write_mapping is None:
            slot_mapping = self._metadata.slot_mapping(
                block_size=config.block_size,
                query_width=query.shape[1],
                device=config.device,
            )
            self._write_mapping = self._cache.prepare_write(slot_mapping, validate=False)
        # 先写入本轮 K/V；因果注意力随后可读取到当前位置自身。
        self._cache.write_prepared(layer_id, key, value, self._write_mapping)

        output = torch.zeros_like(query)
        for row, (table, query_length) in enumerate(
            zip(
                self._metadata.block_tables,
                self._metadata.query_lengths,
                strict=True,
            )
        ):
            logical_positions_by_query = self._metadata.visible_logical_positions(row)
            for query_offset in range(query_length):
                output[row, query_offset] = self._attend_query_token(
                    layer_id,
                    query[row, query_offset],
                    table,
                    logical_positions_by_query[query_offset],
                    scale,
                )
        return output

    def _attend_query_token(
        self,
        layer_id: str,
        query: Tensor,
        block_table: tuple[int, ...],
        logical_positions: tuple[int, ...],
        scale: float,
    ) -> Tensor:
        """逐页读取一个 query 可见的 prefix、祖先和自身 K/V。"""

        layer = self._cache.layer(layer_id)
        layer_spec = self._cache.layer_spec(layer_id)
        block_size = self._cache.config.block_size
        repeats = layer_spec.num_query_heads // layer_spec.num_kv_heads
        accumulator_dtype = (
            torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
        )
        running_max = torch.full(
            (layer_spec.num_query_heads,),
            -torch.inf,
            dtype=accumulator_dtype,
            device=query.device,
        )
        running_sum = torch.zeros_like(running_max)
        running_value = torch.zeros(
            (layer_spec.num_query_heads, layer_spec.head_size),
            dtype=accumulator_dtype,
            device=query.device,
        )
        query_for_math = query.to(accumulator_dtype)

        offsets_by_block: dict[int, list[int]] = {}
        for logical_position in logical_positions:
            logical_block = logical_position // block_size
            offsets_by_block.setdefault(logical_block, []).append(logical_position % block_size)

        # 不拼接完整历史；每次只读取当前 query 真正可见的一页内容。
        for logical_block, offsets in offsets_by_block.items():
            block_id = block_table[logical_block]
            keys = layer.keys[block_id, offsets].to(accumulator_dtype)
            values = layer.values[block_id, offsets].to(accumulator_dtype)
            if repeats > 1:
                keys = keys.repeat_interleave(repeats, dim=1)
                values = values.repeat_interleave(repeats, dim=1)
            keys = keys.permute(1, 0, 2)
            values = values.permute(1, 0, 2)

            scores = torch.einsum("hd,htd->ht", query_for_math, keys) * scale
            block_max = scores.max(dim=1).values
            next_max = torch.maximum(running_max, block_max)
            previous_scale = torch.exp(running_max - next_max)
            weights = torch.exp(scores - next_max.unsqueeze(1))
            running_value = running_value * previous_scale.unsqueeze(1)
            running_value += torch.einsum("ht,htd->hd", weights, values)
            running_sum = running_sum * previous_scale + weights.sum(dim=1)
            running_max = next_max

        return (running_value / running_sum.unsqueeze(1)).to(query.dtype)


class TorchPagedAttentionBackend:
    """创建便于阅读和测试的 PyTorch 分页 attention。"""

    def create(
        self,
        cache: PagedKVCache,
        metadata: PagedAttentionMetadata,
    ) -> TorchPagedAttention:
        return TorchPagedAttention(cache, metadata)
