from __future__ import annotations

from math import sqrt

import pytest
import torch

pytest.importorskip("triton")

from light_vllm.modeling.attention import AttentionLayerSpec, ModelKVCacheSpec
from light_vllm.runtime.execution.interfaces import QueryLayout
from light_vllm.runtime.execution.layout import linear_query_layout
from light_vllm.runtime.execution.paged_attention import (
    PagedAttentionMetadata,
    TorchPagedAttention,
)
from light_vllm.runtime.execution.paged_cache import PagedKVCache, PagedKVCacheConfig
from light_vllm.runtime.execution.triton_paged_attention import TritonPagedAttentionBackend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
_DEVICE = torch.device("cuda:0")


def _layouts(*lengths: int):
    return tuple(linear_query_layout(length) for length in lengths)


def _caches(
    head_size: int,
    dtype: torch.dtype,
    *,
    block_size: int = 2,
) -> tuple[PagedKVCache, PagedKVCache]:
    model_spec = ModelKVCacheSpec(
        (
            AttentionLayerSpec(
                layer_id="attention",
                num_query_heads=4,
                num_kv_heads=2,
                head_size=head_size,
            ),
        )
    )
    config = PagedKVCacheConfig(
        num_blocks=16,
        block_size=block_size,
        dtype=dtype,
        device=_DEVICE,
    )
    return PagedKVCache(model_spec, config), PagedKVCache(model_spec, config)


def _assert_matches_torch(
    torch_cache: PagedKVCache,
    triton_cache: PagedKVCache,
    metadata: PagedAttentionMetadata,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    scale = 1.0 / sqrt(query.shape[-1])
    expected = TorchPagedAttention(torch_cache, metadata).forward(
        "attention",
        query,
        key,
        value,
        scale=scale,
    )
    actual = (
        TritonPagedAttentionBackend()
        .create(triton_cache, metadata)
        .forward(
            "attention",
            query,
            key,
            value,
            scale=scale,
        )
    )
    torch.cuda.synchronize()
    tolerance = 3e-3 if query.dtype == torch.float16 else 2e-2
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("head_size", (32, 80, 128, 256))
@pytest.mark.parametrize(
    "dtype",
    (
        torch.float16,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_bf16_supported(),
                reason="BF16 is not supported by this GPU",
            ),
        ),
    ),
)
def test_triton_paged_attention_matches_packed_gqa_prefill(
    head_size: int,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(20260817)
    torch_cache, triton_cache = _caches(head_size, dtype, block_size=4)
    metadata = PagedAttentionMetadata(
        block_tables=((7, 1), (4,)),
        num_computed_tokens=(0, 0),
        query_layouts=_layouts(6, 3),
    )
    query = torch.randn(9, 4, head_size, device=_DEVICE, dtype=dtype)
    key = torch.randn(9, 2, head_size, device=_DEVICE, dtype=dtype)
    value = torch.randn(9, 2, head_size, device=_DEVICE, dtype=dtype)

    _assert_matches_torch(torch_cache, triton_cache, metadata, query, key, value)


def test_triton_paged_attention_matches_packed_tree_visibility() -> None:
    torch.manual_seed(20260819)
    torch_cache, triton_cache = _caches(64, torch.float16)
    metadata = PagedAttentionMetadata(
        block_tables=((7, 1), (4, 6)),
        num_computed_tokens=(0, 0),
        query_layouts=(
            QueryLayout((-1, 0, 0, 2)),
            QueryLayout((-1, 0, 0)),
        ),
    )
    query = torch.randn(7, 4, 64, device=_DEVICE, dtype=torch.float16)
    key = torch.randn(7, 2, 64, device=_DEVICE, dtype=torch.float16)
    value = torch.randn(7, 2, 64, device=_DEVICE, dtype=torch.float16)

    _assert_matches_torch(torch_cache, triton_cache, metadata, query, key, value)


def test_triton_paged_attention_matches_decode_after_prefill() -> None:
    torch.manual_seed(20260817)
    torch_cache, triton_cache = _caches(64, torch.float16)
    prefill_metadata = PagedAttentionMetadata(
        block_tables=((3, 0, 5), (2, 6)),
        num_computed_tokens=(0, 0),
        query_layouts=_layouts(5, 3),
    )
    prefill_query = torch.randn(8, 4, 64, device=_DEVICE, dtype=torch.float16)
    prefill_key = torch.randn(8, 2, 64, device=_DEVICE, dtype=torch.float16)
    prefill_value = torch.randn(8, 2, 64, device=_DEVICE, dtype=torch.float16)
    _assert_matches_torch(
        torch_cache,
        triton_cache,
        prefill_metadata,
        prefill_query,
        prefill_key,
        prefill_value,
    )

    decode_metadata = PagedAttentionMetadata(
        block_tables=((3, 0, 5), (2, 6)),
        num_computed_tokens=(5, 3),
        query_layouts=_layouts(1, 1),
    )
    decode_query = torch.randn(2, 4, 64, device=_DEVICE, dtype=torch.float16)
    decode_key = torch.randn(2, 2, 64, device=_DEVICE, dtype=torch.float16)
    decode_value = torch.randn(2, 2, 64, device=_DEVICE, dtype=torch.float16)

    _assert_matches_torch(
        torch_cache,
        triton_cache,
        decode_metadata,
        decode_query,
        decode_key,
        decode_value,
    )


def test_triton_paged_attention_reads_a_shared_readonly_prefix() -> None:
    torch.manual_seed(20260817)
    torch_cache, triton_cache = _caches(64, torch.float16)
    prefix_key = torch.randn(2, 2, 64, device=_DEVICE, dtype=torch.float16)
    prefix_value = torch.randn(2, 2, 64, device=_DEVICE, dtype=torch.float16)
    prefix_slots = torch.tensor([0, 1], device=_DEVICE)
    torch_cache.write("attention", prefix_key, prefix_value, prefix_slots)
    triton_cache.write("attention", prefix_key, prefix_value, prefix_slots)
    metadata = PagedAttentionMetadata(
        block_tables=((0, 5), (0, 7)),
        num_computed_tokens=(2, 2),
        query_layouts=_layouts(1, 1),
        num_readonly_prefix_blocks=(1, 1),
    )
    query = torch.randn(2, 4, 64, device=_DEVICE, dtype=torch.float16)
    key = torch.randn(2, 2, 64, device=_DEVICE, dtype=torch.float16)
    value = torch.randn(2, 2, 64, device=_DEVICE, dtype=torch.float16)

    _assert_matches_torch(torch_cache, triton_cache, metadata, query, key, value)


def test_triton_paged_attention_ignores_unused_lookahead_slots() -> None:
    torch.manual_seed(20260817)
    torch_cache, triton_cache = _caches(64, torch.float16)
    metadata = PagedAttentionMetadata(
        block_tables=((8, 3, 12), (10, 14)),
        num_computed_tokens=(0, 0),
        query_layouts=_layouts(3, 1),
        num_reserved_query_tokens=(5, 4),
    )
    query = torch.randn(4, 4, 64, device=_DEVICE, dtype=torch.float16)
    key = torch.randn(4, 2, 64, device=_DEVICE, dtype=torch.float16)
    value = torch.randn(4, 2, 64, device=_DEVICE, dtype=torch.float16)

    _assert_matches_torch(torch_cache, triton_cache, metadata, query, key, value)
