"""可读性优先的 PyTorch Paged Attention 正确性实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from light_vllm.modeling.attention.interfaces import AttentionContext
from light_vllm.runtime.execution.paged_cache import PagedKVCache
from light_vllm.runtime.kv_cache import KVCacheError


@dataclass(frozen=True, slots=True)
class PagedAttentionMetadata:
    """一个 padded query batch 访问分页 K/V 所需的事实。"""

    block_tables: tuple[tuple[int, ...], ...]
    num_computed_tokens: tuple[int, ...]
    query_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        block_tables = tuple(tuple(table) for table in self.block_tables)
        computed = tuple(self.num_computed_tokens)
        query_lengths = tuple(self.query_lengths)
        num_requests = len(block_tables)
        if num_requests == 0:
            raise ValueError("paged attention metadata must contain at least one request")
        if len(computed) != num_requests or len(query_lengths) != num_requests:
            raise ValueError("paged attention metadata fields must have the same batch size")
        if any(type(value) is not int or value < 0 for value in computed):
            raise ValueError("computed-token counts must be non-negative integers")
        if any(type(value) is not int or value <= 0 for value in query_lengths):
            raise ValueError("query lengths must be positive integers")
        if any(
            type(block_id) is not int or block_id < 0
            for table in block_tables
            for block_id in table
        ):
            raise ValueError("block tables must contain non-negative integers")
        object.__setattr__(self, "block_tables", block_tables)
        object.__setattr__(self, "num_computed_tokens", computed)
        object.__setattr__(self, "query_lengths", query_lengths)

    @property
    def batch_size(self) -> int:
        return len(self.block_tables)

    def slot_mapping(
        self,
        *,
        block_size: int,
        query_width: int,
        device: torch.device,
    ) -> Tensor:
        """把每个 query token 的逻辑位置转换成物理 cache slot。"""

        mapping = torch.full(
            (self.batch_size, query_width),
            -1,
            dtype=torch.long,
            device=device,
        )
        for row, (table, computed, query_length) in enumerate(
            zip(
                self.block_tables,
                self.num_computed_tokens,
                self.query_lengths,
                strict=True,
            )
        ):
            if query_length > query_width:
                raise KVCacheError("query length exceeds the padded query width")
            total_tokens = computed + query_length
            required_blocks = (total_tokens + block_size - 1) // block_size
            if len(table) < required_blocks:
                raise KVCacheError("block table does not cover all scheduled tokens")
            if len(table) > required_blocks:
                raise KVCacheError("block table contains unused physical blocks")
            for query_offset in range(query_length):
                position = computed + query_offset
                block_id = table[position // block_size]
                mapping[row, query_offset] = block_id * block_size + position % block_size
        return mapping

    def validate_block_tables(
        self,
        *,
        num_blocks: int,
        block_size: int,
    ) -> None:
        """校验本轮可能读取的完整 block table 都落在物理页池内。"""

        owned_blocks: set[int] = set()
        for table, computed, query_length in zip(
            self.block_tables,
            self.num_computed_tokens,
            self.query_lengths,
            strict=True,
        ):
            required_blocks = (computed + query_length + block_size - 1) // block_size
            if len(table) < required_blocks:
                raise KVCacheError("block table does not cover all scheduled tokens")
            if len(table) > required_blocks:
                raise KVCacheError("block table contains unused physical blocks")
            if any(block_id >= num_blocks for block_id in table):
                raise KVCacheError("block table contains an out-of-range physical block")
            if len(set(table)) != len(table):
                raise KVCacheError("block table aliases a physical block within one request")
            if owned_blocks.intersection(table):
                # Prefix sharing 需要显式的只读 ownership；当前所有活动页必须独占。
                raise KVCacheError("block tables alias a physical block across requests")
            owned_blocks.update(table)


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


class TorchPagedAttention:
    """逐物理页读取 K/V 的在线 softmax attention。

    这是便于阅读和测试的 PyTorch 参考实现。它不会先拼接完整历史 K/V，
    而是边读取每一页边累计 softmax。以后 CUDA/Triton 实现继续使用同一接口。
    """

    def __init__(self, cache: PagedKVCache, metadata: PagedAttentionMetadata) -> None:
        self._cache = cache
        self._metadata = metadata
        self._layer_ids: set[str] = set()

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
        layer_spec = self._cache.layer_spec(layer_id)
        config = self._cache.config
        if query.ndim != 4:
            raise KVCacheError("paged attention query must have four dimensions")
        if query.shape[:2] != key.shape[:2] or value.shape != key.shape:
            raise KVCacheError("paged attention Q/K/V batch and query dimensions must match")
        if query.shape[0] != self._metadata.batch_size:
            raise KVCacheError("paged attention metadata batch size does not match query")
        if query.shape[2:] != (layer_spec.num_query_heads, layer_spec.head_size):
            raise KVCacheError("paged attention query shape does not match the layer spec")
        if key.shape[2:] != (layer_spec.num_kv_heads, layer_spec.head_size):
            raise KVCacheError("paged attention K/V shape does not match the layer spec")
        self._metadata.validate_block_tables(
            num_blocks=config.num_blocks,
            block_size=config.block_size,
        )

        slot_mapping = self._metadata.slot_mapping(
            block_size=config.block_size,
            query_width=query.shape[1],
            device=config.device,
        )
        # 先写入本轮 K/V；因果注意力随后可读取到当前位置自身。
        self._cache.write(layer_id, key, value, slot_mapping)

        output = torch.zeros_like(query)
        for row, (table, computed, query_length) in enumerate(
            zip(
                self._metadata.block_tables,
                self._metadata.num_computed_tokens,
                self._metadata.query_lengths,
                strict=True,
            )
        ):
            for query_offset in range(query_length):
                sequence_length = computed + query_offset + 1
                output[row, query_offset] = self._attend_one(
                    layer_id,
                    query[row, query_offset],
                    table,
                    sequence_length,
                    scale,
                )
        return output

    def _attend_one(
        self,
        layer_id: str,
        query: Tensor,
        block_table: tuple[int, ...],
        sequence_length: int,
        scale: float,
    ) -> Tensor:
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

        # 每读一页就更新 softmax 的累计值，不需要先拼出完整历史 K/V。
        num_blocks = (sequence_length + block_size - 1) // block_size
        for logical_block in range(num_blocks):
            block_id = block_table[logical_block]
            tokens_in_block = min(block_size, sequence_length - logical_block * block_size)
            keys = layer.keys[block_id, :tokens_in_block].to(accumulator_dtype)
            values = layer.values[block_id, :tokens_in_block].to(accumulator_dtype)
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
