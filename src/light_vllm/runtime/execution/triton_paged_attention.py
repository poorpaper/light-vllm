"""直接读取分页 K/V 的 Triton attention backend。"""

from __future__ import annotations

import torch
from torch import Tensor

from light_vllm.runtime.execution.interfaces import ExecutionNotReadyError
from light_vllm.runtime.execution.paged_attention import (
    PagedAttentionMetadata,
    _validate_paged_attention_tensors,
)
from light_vllm.runtime.execution.paged_cache import PagedKVCache, PagedKVWriteMapping
from light_vllm.runtime.kv_cache import KVCacheError

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # Windows/CPU 环境仍应能正常导入整个项目。
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR: ImportError | None = exc
else:
    _TRITON_IMPORT_ERROR = None


if triton is not None:

    @triton.jit
    def _paged_attention_kernel(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_table_ptr,
        computed_ptr,
        query_length_ptr,
        query_visibility_ptr,
        output_ptr,
        stride_query_batch,
        stride_query_token,
        stride_query_head,
        stride_query_dim,
        stride_key_block,
        stride_key_token,
        stride_key_head,
        stride_key_dim,
        stride_value_block,
        stride_value_token,
        stride_value_head,
        stride_value_dim,
        stride_table_batch,
        stride_visibility_batch,
        stride_visibility_query,
        stride_visibility_key,
        stride_output_batch,
        stride_output_token,
        stride_output_head,
        stride_output_dim,
        max_sequence_length,
        scale,
        LINEAR_QUERY_LAYOUTS: tl.constexpr,
        QUERY_WIDTH: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """一个 program 计算一个 query token 的一个 query head。"""

        query_index = tl.program_id(0)
        query_head = tl.program_id(1)
        batch_index = query_index // QUERY_WIDTH
        query_offset = query_index % QUERY_WIDTH
        query_length = tl.load(query_length_ptr + batch_index)
        active_query = query_offset < query_length
        computed = tl.load(computed_ptr + batch_index)
        kv_head = query_head // GROUP_SIZE

        dim_offsets = tl.arange(0, BLOCK_D)
        dim_mask = dim_offsets < HEAD_SIZE
        query_offsets = (
            batch_index * stride_query_batch
            + query_offset * stride_query_token
            + query_head * stride_query_head
            + dim_offsets * stride_query_dim
        )
        query = tl.load(
            query_ptr + query_offsets,
            mask=active_query & dim_mask,
            other=0.0,
        ).to(tl.float32)

        running_max = -1.0e20
        running_sum = 0.0
        running_value = tl.zeros((BLOCK_D,), dtype=tl.float32)
        token_offsets = tl.arange(0, BLOCK_N)

        # 每次读一小段分页历史，并在线更新 softmax，避免创建中间分数张量。
        for block_start in tl.range(0, max_sequence_length, BLOCK_N):
            positions = block_start + token_offsets
            key_query_offsets = positions - computed
            query_key_mask = (key_query_offsets >= 0) & (key_query_offsets < query_length)
            if LINEAR_QUERY_LAYOUTS:
                query_key_visible = query_key_mask & (key_query_offsets <= query_offset)
            else:
                safe_key_query_offsets = tl.maximum(key_query_offsets, 0)
                visibility_offsets = (
                    batch_index * stride_visibility_batch
                    + query_offset * stride_visibility_query
                    + safe_key_query_offsets * stride_visibility_key
                )
                query_key_visible = tl.load(
                    query_visibility_ptr + visibility_offsets,
                    mask=active_query & query_key_mask,
                    other=0,
                ).to(tl.int1)
            # committed prefix 全部可见；本轮 query 只读取父链上的祖先和自身。
            token_mask = active_query & ((positions < computed) | query_key_visible)
            logical_blocks = positions // PAGE_SIZE
            offsets_in_page = positions % PAGE_SIZE
            physical_blocks = tl.load(
                block_table_ptr + batch_index * stride_table_batch + logical_blocks,
                mask=token_mask,
                other=0,
            )

            key_offsets = (
                physical_blocks[:, None] * stride_key_block
                + offsets_in_page[:, None] * stride_key_token
                + kv_head * stride_key_head
                + dim_offsets[None, :] * stride_key_dim
            )
            keys = tl.load(
                key_cache_ptr + key_offsets,
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * query[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, -float("inf"))

            block_max = tl.max(scores, axis=0)
            next_max = tl.maximum(running_max, block_max)
            previous_scale = tl.exp(running_max - next_max)
            weights = tl.exp(scores - next_max)

            value_offsets = (
                physical_blocks[:, None] * stride_value_block
                + offsets_in_page[:, None] * stride_value_token
                + kv_head * stride_value_head
                + dim_offsets[None, :] * stride_value_dim
            )
            values = tl.load(
                value_cache_ptr + value_offsets,
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            running_value = running_value * previous_scale
            running_value += tl.sum(weights[:, None] * values, axis=0)
            running_sum = running_sum * previous_scale + tl.sum(weights, axis=0)
            running_max = next_max

        output_offsets = (
            batch_index * stride_output_batch
            + query_offset * stride_output_token
            + query_head * stride_output_head
            + dim_offsets * stride_output_dim
        )
        tl.store(
            output_ptr + output_offsets,
            tl.where(active_query, running_value / running_sum, 0.0),
            mask=dim_mask,
        )

else:
    _paged_attention_kernel = None


class TritonPagedAttention:
    """先写本轮 K/V，再用 Triton 融合完成分页 attention。"""

    def __init__(self, cache: PagedKVCache, metadata: PagedAttentionMetadata) -> None:
        self._cache = cache
        self._metadata = metadata
        self._layer_ids: set[str] = set()
        self._slot_mapping: Tensor | None = None
        self._write_mapping: PagedKVWriteMapping | None = None
        self._query_visibility: Tensor | None = None
        self._query_width: int | None = None
        self._linear_query_layouts = metadata.has_only_linear_queries

        config = cache.config
        if config.device.type != "cuda":
            raise KVCacheError("Triton paged attention requires a CUDA KV cache")
        metadata.validate_block_tables(
            num_blocks=config.num_blocks,
            block_size=config.block_size,
        )
        max_blocks = max(len(table) for table in metadata.block_tables)
        padded_tables = tuple(
            table + (0,) * (max_blocks - len(table)) for table in metadata.block_tables
        )
        # 这些 batch 事实供所有 attention 层复用，不在每层重复创建 GPU tensor。
        self._block_tables = torch.tensor(
            padded_tables,
            dtype=torch.int32,
            device=config.device,
        )
        self._num_computed_tokens = torch.tensor(
            metadata.num_computed_tokens,
            dtype=torch.int32,
            device=config.device,
        )
        self._query_lengths = torch.tensor(
            metadata.query_lengths,
            dtype=torch.int32,
            device=config.device,
        )
        self._max_sequence_length = max(
            computed + query_length
            for computed, query_length in zip(
                metadata.num_computed_tokens,
                metadata.query_lengths,
                strict=True,
            )
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
        layer_spec = _validate_paged_attention_tensors(
            self._cache,
            self._metadata,
            layer_id,
            query,
            key,
            value,
        )
        config = self._cache.config
        if query.device != config.device or query.dtype != config.dtype:
            raise KVCacheError("Triton query dtype and device must match the KV cache")
        if layer_spec.head_size > 256:
            raise KVCacheError("Triton paged attention supports head sizes up to 256")

        query_width = query.shape[1]
        if self._query_width not in (None, query_width):
            raise KVCacheError("all Triton attention layers must use the same query width")
        self._query_width = query_width
        if self._slot_mapping is None:
            self._slot_mapping = self._metadata.slot_mapping(
                block_size=config.block_size,
                query_width=query_width,
                device=config.device,
            )
            if not self._linear_query_layouts:
                self._query_visibility = self._metadata.visibility_tensor(
                    query_width=query_width,
                    device=config.device,
                )
            # validate_block_tables() in the constructor already proves that
            # generated writable slots are in range and do not alias.
            self._write_mapping = self._cache.prepare_write(
                self._slot_mapping,
                validate=False,
            )
        assert self._write_mapping is not None

        query_visibility = self._query_visibility
        if query_visibility is None:
            # The linear kernel specializes away every visibility load. A
            # three-dimensional view only supplies the otherwise unused pointer/strides.
            query_visibility = query[..., 0]

        # KV 写入和 attention 分成两个顺序步骤；同一 CUDA stream 保证读取前写入完成。
        layer = self._cache.layer(layer_id)
        block_d = triton.next_power_of_2(layer_spec.head_size)
        self._cache.write_prepared(layer_id, key, value, self._write_mapping)
        output = torch.empty_like(query)
        block_n = min(
            256,
            max(64, triton.next_power_of_2(self._max_sequence_length)),
        )
        num_warps = 4
        grid = (query.shape[0] * query_width, layer_spec.num_query_heads)
        _paged_attention_kernel[grid](
            query,
            layer.keys,
            layer.values,
            self._block_tables,
            self._num_computed_tokens,
            self._query_lengths,
            query_visibility,
            output,
            *query.stride(),
            *layer.keys.stride(),
            *layer.values.stride(),
            self._block_tables.stride(0),
            *query_visibility.stride(),
            *output.stride(),
            self._max_sequence_length,
            scale,
            LINEAR_QUERY_LAYOUTS=self._linear_query_layouts,
            QUERY_WIDTH=query_width,
            PAGE_SIZE=config.block_size,
            GROUP_SIZE=layer_spec.num_query_heads // layer_spec.num_kv_heads,
            HEAD_SIZE=layer_spec.head_size,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        )
        return output


class TritonPagedAttentionBackend:
    """创建可选的 Triton 分页 attention；未安装依赖时给出明确错误。"""

    def __init__(self) -> None:
        if _TRITON_IMPORT_ERROR is not None:
            raise ExecutionNotReadyError(
                "install the optional Triton dependency before selecting this backend"
            ) from _TRITON_IMPORT_ERROR

    def create(
        self,
        cache: PagedKVCache,
        metadata: PagedAttentionMetadata,
    ) -> TritonPagedAttention:
        return TritonPagedAttention(cache, metadata)
