"""直接读取分页 K/V 的 Triton attention backend。"""

from __future__ import annotations

import torch
from torch import Tensor

from light_vllm.runtime.execution.cuda_staging import SingleStepCudaStagingBuffer
from light_vllm.runtime.execution.interfaces import ExecutionNotReadyError
from light_vllm.runtime.execution.paged_attention import (
    PagedAttentionMetadata,
    _validate_paged_attention_tensors,
)
from light_vllm.runtime.execution.paged_cache import PagedKVCache
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
    def _write_paged_kv_kernel(
        key_ptr,
        value_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_table_ptr,
        computed_ptr,
        query_start_loc_ptr,
        request_indices_ptr,
        stride_key_token,
        stride_key_head,
        stride_key_dim,
        stride_value_token,
        stride_value_head,
        stride_value_dim,
        stride_key_cache_block,
        stride_key_cache_token,
        stride_key_cache_head,
        stride_key_cache_dim,
        stride_value_cache_block,
        stride_value_cache_token,
        stride_value_cache_head,
        stride_value_cache_dim,
        stride_table_batch,
        PAGE_SIZE: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """一个 program 同时写一个 token/head 的 K 和 V。"""

        token_index = tl.program_id(0)
        kv_head = tl.program_id(1)
        batch_index = tl.load(request_indices_ptr + token_index)
        query_offset = token_index - tl.load(query_start_loc_ptr + batch_index)
        logical_position = tl.load(computed_ptr + batch_index) + query_offset
        physical_block = tl.load(
            block_table_ptr + batch_index * stride_table_batch + logical_position // PAGE_SIZE
        )
        offset_in_page = logical_position % PAGE_SIZE
        dim_offsets = tl.arange(0, BLOCK_D)
        dim_mask = dim_offsets < HEAD_SIZE

        key_offsets = (
            token_index * stride_key_token
            + kv_head * stride_key_head
            + dim_offsets * stride_key_dim
        )
        value_offsets = (
            token_index * stride_value_token
            + kv_head * stride_value_head
            + dim_offsets * stride_value_dim
        )
        key_cache_offsets = (
            physical_block * stride_key_cache_block
            + offset_in_page * stride_key_cache_token
            + kv_head * stride_key_cache_head
            + dim_offsets * stride_key_cache_dim
        )
        value_cache_offsets = (
            physical_block * stride_value_cache_block
            + offset_in_page * stride_value_cache_token
            + kv_head * stride_value_cache_head
            + dim_offsets * stride_value_cache_dim
        )
        tl.store(
            key_cache_ptr + key_cache_offsets,
            tl.load(key_ptr + key_offsets, mask=dim_mask),
            mask=dim_mask,
        )
        tl.store(
            value_cache_ptr + value_cache_offsets,
            tl.load(value_ptr + value_offsets, mask=dim_mask),
            mask=dim_mask,
        )

    @triton.jit
    def _paged_attention_kernel(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_table_ptr,
        computed_ptr,
        query_start_loc_ptr,
        request_indices_ptr,
        query_length_ptr,
        query_visibility_ptr,
        visibility_start_ptr,
        output_ptr,
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
        stride_output_token,
        stride_output_head,
        stride_output_dim,
        max_sequence_length,
        scale,
        LINEAR_QUERY_LAYOUTS: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        BLOCK_G: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """一个 program 计算一个 query token 的一个 query head。"""

        token_index = tl.program_id(0)
        query_head = tl.program_id(1)
        batch_index = tl.load(request_indices_ptr + token_index)
        query_offset = token_index - tl.load(query_start_loc_ptr + batch_index)
        query_length = tl.load(query_length_ptr + batch_index)
        computed = tl.load(computed_ptr + batch_index)
        kv_head = query_head // GROUP_SIZE

        dim_offsets = tl.arange(0, BLOCK_D)
        dim_mask = dim_offsets < HEAD_SIZE
        query_offsets = (
            token_index * stride_query_token
            + query_head * stride_query_head
            + dim_offsets * stride_query_dim
        )
        query = tl.load(
            query_ptr + query_offsets,
            mask=dim_mask,
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
                    tl.load(visibility_start_ptr + batch_index)
                    + query_offset * query_length
                    + safe_key_query_offsets
                )
                query_key_visible = tl.load(
                    query_visibility_ptr + visibility_offsets,
                    mask=query_key_mask,
                    other=0,
                ).to(tl.int1)
            # committed prefix 全部可见；本轮 query 只读取父链上的祖先和自身。
            token_mask = (positions < computed) | query_key_visible
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
            token_index * stride_output_token
            + query_head * stride_output_head
            + dim_offsets * stride_output_dim
        )
        tl.store(
            output_ptr + output_offsets,
            running_value / running_sum,
            mask=dim_mask,
        )

    @triton.jit
    def _grouped_paged_attention_kernel(
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_table_ptr,
        computed_ptr,
        query_start_loc_ptr,
        request_indices_ptr,
        query_length_ptr,
        query_visibility_ptr,
        visibility_start_ptr,
        output_ptr,
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
        stride_output_token,
        stride_output_head,
        stride_output_dim,
        max_sequence_length,
        scale,
        LINEAR_QUERY_LAYOUTS: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        BLOCK_G: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """一个 program 复用 K/V，计算同组的全部 query heads。"""

        token_index = tl.program_id(0)
        kv_head = tl.program_id(1)
        batch_index = tl.load(request_indices_ptr + token_index)
        query_offset = token_index - tl.load(query_start_loc_ptr + batch_index)
        query_length = tl.load(query_length_ptr + batch_index)
        computed = tl.load(computed_ptr + batch_index)

        group_offsets = tl.arange(0, BLOCK_G)
        dim_offsets = tl.arange(0, BLOCK_D)
        group_mask = group_offsets < GROUP_SIZE
        dim_mask = dim_offsets < HEAD_SIZE
        query_heads = kv_head * GROUP_SIZE + group_offsets
        query_offsets = (
            token_index * stride_query_token
            + query_heads[None, :] * stride_query_head
            + dim_offsets[:, None] * stride_query_dim
        )
        queries = tl.load(
            query_ptr + query_offsets,
            mask=dim_mask[:, None] & group_mask[None, :],
            other=0.0,
        )

        running_max = tl.full((BLOCK_G,), -1.0e20, tl.float32)
        running_sum = tl.zeros((BLOCK_G,), dtype=tl.float32)
        running_value = tl.zeros((BLOCK_G, BLOCK_D), dtype=tl.float32)
        token_offsets = tl.arange(0, BLOCK_N)

        # GQA 的一组 query heads 共用同一份 K/V；矩阵乘一次完成整组计算。
        for block_start in tl.range(0, max_sequence_length, BLOCK_N):
            positions = block_start + token_offsets
            key_query_offsets = positions - computed
            query_key_mask = (key_query_offsets >= 0) & (key_query_offsets < query_length)
            if LINEAR_QUERY_LAYOUTS:
                query_key_visible = query_key_mask & (key_query_offsets <= query_offset)
            else:
                safe_key_query_offsets = tl.maximum(key_query_offsets, 0)
                visibility_offsets = (
                    tl.load(visibility_start_ptr + batch_index)
                    + query_offset * query_length
                    + safe_key_query_offsets
                )
                query_key_visible = tl.load(
                    query_visibility_ptr + visibility_offsets,
                    mask=query_key_mask,
                    other=0,
                ).to(tl.int1)
            token_mask = (positions < computed) | query_key_visible
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
            )
            scores = tl.dot(keys, queries) * scale
            scores = tl.where(
                token_mask[:, None] & group_mask[None, :],
                scores,
                -float("inf"),
            )

            block_max = tl.max(scores, axis=0)
            next_max = tl.maximum(running_max, block_max)
            previous_scale = tl.exp(running_max - next_max)
            weights = tl.exp(scores - next_max[None, :])

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
            )
            running_value *= previous_scale[:, None]
            running_value += tl.dot(tl.trans(weights).to(queries.dtype), values)
            running_sum = running_sum * previous_scale + tl.sum(weights, axis=0)
            running_max = next_max

        output_offsets = (
            token_index * stride_output_token
            + query_heads[:, None] * stride_output_head
            + dim_offsets[None, :] * stride_output_dim
        )
        tl.store(
            output_ptr + output_offsets,
            running_value / running_sum[:, None],
            mask=group_mask[:, None] & dim_mask[None, :],
        )

else:
    _write_paged_kv_kernel = None
    _paged_attention_kernel = None
    _grouped_paged_attention_kernel = None


class TritonPagedAttention:
    """先写本轮 K/V，再用 Triton 融合完成分页 attention。"""

    def __init__(
        self,
        cache: PagedKVCache,
        metadata: PagedAttentionMetadata,
        staging: SingleStepCudaStagingBuffer,
    ) -> None:
        self._cache = cache
        self._metadata = metadata
        self._layer_ids: set[str] = set()
        self._query_visibility: Tensor | None = None
        self._visibility_start: Tensor | None = None
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
        # 多组小 metadata 共用 pinned staging，一次 H2D 后由所有层只读。
        (
            flat_block_tables,
            self._num_computed_tokens,
            self._query_lengths,
            self._query_start_loc,
            self._request_indices,
        ) = staging.copy_groups(
            tuple(value for table in padded_tables for value in table),
            metadata.num_computed_tokens,
            metadata.query_lengths,
            metadata.query_start_loc,
            metadata.request_indices,
        )
        self._block_tables = flat_block_tables.view(metadata.batch_size, max_blocks)
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

        if not self._linear_query_layouts and self._query_visibility is None:
            self._query_visibility, self._visibility_start = self._metadata.visibility_tensor(
                device=config.device
            )

        query_visibility = self._query_visibility
        if query_visibility is None:
            # 线性 kernel 会编译掉 visibility 读取，这里只提供未使用的合法指针。
            query_visibility = query[..., 0]
        visibility_start = self._visibility_start
        if visibility_start is None:
            visibility_start = self._query_start_loc

        layer = self._cache.layer(layer_id)
        block_d = triton.next_power_of_2(layer_spec.head_size)
        # 一个 KV head 同时服务整组 Q heads，避免 GQA 重复读取同一段 K/V。
        group_size = layer_spec.num_query_heads // layer_spec.num_kv_heads
        use_grouped_kernel = group_size > 1 and group_size <= 16 and block_d <= 128
        write_grid = (key.shape[0], layer_spec.num_kv_heads)
        _write_paged_kv_kernel[write_grid](
            key,
            value,
            layer.keys,
            layer.values,
            self._block_tables,
            self._num_computed_tokens,
            self._query_start_loc,
            self._request_indices,
            *key.stride(),
            *value.stride(),
            *layer.keys.stride(),
            *layer.values.stride(),
            self._block_tables.stride(0),
            PAGE_SIZE=config.block_size,
            HEAD_SIZE=layer_spec.head_size,
            BLOCK_D=block_d,
            num_warps=4,
        )

        output = torch.empty_like(query)
        block_n = min(
            256,
            max(64, triton.next_power_of_2(self._max_sequence_length)),
        )
        num_warps = 4
        kernel = _grouped_paged_attention_kernel if use_grouped_kernel else _paged_attention_kernel
        kernel_block_n = min(block_n, 64) if use_grouped_kernel else block_n
        grid = (
            query.shape[0],
            layer_spec.num_kv_heads if use_grouped_kernel else layer_spec.num_query_heads,
        )
        kernel[grid](
            query,
            layer.keys,
            layer.values,
            self._block_tables,
            self._num_computed_tokens,
            self._query_start_loc,
            self._request_indices,
            self._query_lengths,
            query_visibility,
            visibility_start,
            output,
            *query.stride(),
            *layer.keys.stride(),
            *layer.values.stride(),
            self._block_tables.stride(0),
            *output.stride(),
            self._max_sequence_length,
            scale,
            LINEAR_QUERY_LAYOUTS=self._linear_query_layouts,
            PAGE_SIZE=config.block_size,
            GROUP_SIZE=group_size,
            HEAD_SIZE=layer_spec.head_size,
            BLOCK_G=max(16, triton.next_power_of_2(group_size)) if use_grouped_kernel else 1,
            BLOCK_D=block_d,
            BLOCK_N=kernel_block_n,
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
        self._staging: SingleStepCudaStagingBuffer | None = None

    def create(
        self,
        cache: PagedKVCache,
        metadata: PagedAttentionMetadata,
    ) -> TritonPagedAttention:
        device = cache.config.device
        if self._staging is None:
            self._staging = SingleStepCudaStagingBuffer(device, torch.int32)
        return TritonPagedAttention(cache, metadata, self._staging)
