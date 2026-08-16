from __future__ import annotations

from math import sqrt

import pytest
import torch

from light_vllm.modeling.attention import AttentionLayerSpec, ModelKVCacheSpec
from light_vllm.runtime.execution.paged_attention import (
    PagedAttentionMetadata,
    TorchPagedAttention,
)
from light_vllm.runtime.execution.paged_cache import (
    CudaMemoryKVCachePlanner,
    PagedKVCache,
    PagedKVCacheConfig,
    kv_cache_bytes_per_block,
)
from light_vllm.runtime.kv_cache import KVCacheError, PagedKVCacheManager


def _cache(*, num_query_heads: int = 2, num_kv_heads: int = 2) -> PagedKVCache:
    return PagedKVCache(
        ModelKVCacheSpec(
            layers=(
                AttentionLayerSpec(
                    layer_id="attention",
                    num_query_heads=num_query_heads,
                    num_kv_heads=num_kv_heads,
                    head_size=4,
                ),
            )
        ),
        PagedKVCacheConfig(num_blocks=4, block_size=2),
    )


def test_cuda_memory_planner_derives_capacity_from_the_model_kv_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_spec = _cache().model_spec
    planner = CudaMemoryKVCachePlanner(
        block_size=2,
        dtype=torch.float32,
        device="cuda:0",
        memory_fraction=0.5,
    )
    logical_cache = PagedKVCacheManager(planner)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1024, 2048))

    with pytest.raises(KVCacheError, match="before planning"):
        _ = logical_cache.num_free_blocks
    config = planner.plan(model_spec)

    assert kv_cache_bytes_per_block(model_spec, block_size=2, dtype=torch.float32) == 128
    assert config.num_blocks == 4
    assert planner.num_blocks == 4
    assert logical_cache.num_free_blocks == 4


def _dense_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    repeats = query.shape[0] // keys.shape[1]
    if repeats > 1:
        keys = keys.repeat_interleave(repeats, dim=1)
        values = values.repeat_interleave(repeats, dim=1)
    scores = torch.einsum("hd,thd->ht", query, keys) / sqrt(query.shape[-1])
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("ht,thd->hd", probabilities, values)


def test_slot_mapping_uses_non_contiguous_physical_blocks() -> None:
    metadata = PagedAttentionMetadata(
        block_tables=((2, 0), (1, 3)),
        num_computed_tokens=(0, 1),
        query_lengths=(3, 2),
    )

    mapping = metadata.slot_mapping(block_size=2, query_width=3, device=torch.device("cpu"))

    assert mapping.tolist() == [[4, 5, 0], [3, 6, -1]]


def test_paged_attention_matches_dense_attention_across_blocks_and_requests() -> None:
    torch.manual_seed(13)
    cache = _cache()
    first_metadata = PagedAttentionMetadata(
        block_tables=((2, 0), (1,)),
        num_computed_tokens=(0, 0),
        query_lengths=(3, 2),
    )
    first_query = torch.randn(2, 3, 2, 4)
    first_key = torch.randn(2, 3, 2, 4)
    first_value = torch.randn(2, 3, 2, 4)

    first_output = TorchPagedAttention(cache, first_metadata).forward(
        "attention",
        first_query,
        first_key,
        first_value,
        scale=0.5,
    )

    for row, query_length in enumerate((3, 2)):
        for offset in range(query_length):
            expected = _dense_attention(
                first_query[row, offset],
                first_key[row, : offset + 1],
                first_value[row, : offset + 1],
            )
            torch.testing.assert_close(first_output[row, offset], expected)
    assert first_output[1, 2].count_nonzero() == 0

    decode_metadata = PagedAttentionMetadata(
        block_tables=((2, 0), (1, 3)),
        num_computed_tokens=(3, 2),
        query_lengths=(1, 1),
    )
    decode_query = torch.randn(2, 1, 2, 4)
    decode_key = torch.randn(2, 1, 2, 4)
    decode_value = torch.randn(2, 1, 2, 4)

    decode_output = TorchPagedAttention(cache, decode_metadata).forward(
        "attention",
        decode_query,
        decode_key,
        decode_value,
        scale=0.5,
    )

    for row, previous_length in enumerate((3, 2)):
        keys = torch.cat((first_key[row, :previous_length], decode_key[row]), dim=0)
        values = torch.cat((first_value[row, :previous_length], decode_value[row]), dim=0)
        expected = _dense_attention(decode_query[row, 0], keys, values)
        torch.testing.assert_close(decode_output[row, 0], expected)


def test_paged_attention_supports_grouped_query_heads() -> None:
    torch.manual_seed(17)
    cache = _cache(num_query_heads=4, num_kv_heads=2)
    metadata = PagedAttentionMetadata(
        block_tables=((3, 0),),
        num_computed_tokens=(0,),
        query_lengths=(3,),
    )
    query = torch.randn(1, 3, 4, 4)
    key = torch.randn(1, 3, 2, 4)
    value = torch.randn(1, 3, 2, 4)

    output = TorchPagedAttention(cache, metadata).forward(
        "attention",
        query,
        key,
        value,
        scale=0.5,
    )

    for offset in range(3):
        expected = _dense_attention(query[0, offset], key[0, : offset + 1], value[0, : offset + 1])
        torch.testing.assert_close(output[0, offset], expected)


def test_slot_mapping_rejects_a_block_table_that_does_not_cover_the_query() -> None:
    metadata = PagedAttentionMetadata(
        block_tables=((0,),),
        num_computed_tokens=(1,),
        query_lengths=(2,),
    )

    with pytest.raises(KVCacheError, match="block table does not cover"):
        metadata.slot_mapping(block_size=2, query_width=2, device=torch.device("cpu"))


def test_paged_attention_rejects_an_out_of_range_history_block() -> None:
    cache = _cache()
    metadata = PagedAttentionMetadata(
        # 当前 token 写入 block 0，但 attention 也会读取越界的历史 block 4。
        block_tables=((4, 0),),
        num_computed_tokens=(2,),
        query_lengths=(1,),
    )

    with pytest.raises(KVCacheError, match="out-of-range physical block"):
        TorchPagedAttention(cache, metadata).forward(
            "attention",
            torch.randn(1, 1, 2, 4),
            torch.randn(1, 1, 2, 4),
            torch.randn(1, 1, 2, 4),
            scale=0.5,
        )


def test_paged_attention_rejects_an_aliased_block_within_one_request() -> None:
    cache = _cache()
    metadata = PagedAttentionMetadata(
        block_tables=((0, 0),),
        num_computed_tokens=(2,),
        query_lengths=(1,),
    )

    with pytest.raises(KVCacheError, match="aliases a physical block within one request"):
        TorchPagedAttention(cache, metadata).forward(
            "attention",
            torch.randn(1, 1, 2, 4),
            torch.randn(1, 1, 2, 4),
            torch.randn(1, 1, 2, 4),
            scale=0.5,
        )


def test_paged_attention_rejects_aliased_blocks_across_requests() -> None:
    cache = _cache()
    metadata = PagedAttentionMetadata(
        # 两行写不同 offset，单靠重复 slot 校验无法发现它们共享同一物理页。
        block_tables=((0,), (0,)),
        num_computed_tokens=(0, 1),
        query_lengths=(1, 1),
    )

    with pytest.raises(KVCacheError, match="alias a physical block across requests"):
        TorchPagedAttention(cache, metadata).forward(
            "attention",
            torch.randn(2, 1, 2, 4),
            torch.randn(2, 1, 2, 4),
            torch.randn(2, 1, 2, 4),
            scale=0.5,
        )


def test_paged_attention_allows_the_same_readonly_prefix_position() -> None:
    torch.manual_seed(29)
    cache = _cache()
    past_keys = torch.randn(1, 2, 2, 4)
    past_values = torch.randn(1, 2, 2, 4)
    cache.write("attention", past_keys, past_values, torch.tensor([[0, 1]]))
    metadata = PagedAttentionMetadata(
        block_tables=((0, 1), (0, 2)),
        num_computed_tokens=(2, 2),
        query_lengths=(1, 1),
        num_readonly_prefix_blocks=(1, 1),
    )
    query = torch.randn(2, 1, 2, 4)
    key = torch.randn(2, 1, 2, 4)
    value = torch.randn(2, 1, 2, 4)

    output = TorchPagedAttention(cache, metadata).forward(
        "attention",
        query,
        key,
        value,
        scale=0.5,
    )

    for row in range(2):
        expected = _dense_attention(
            query[row, 0],
            torch.cat((past_keys[0], key[row]), dim=0),
            torch.cat((past_values[0], value[row]), dim=0),
        )
        torch.testing.assert_close(output[row, 0], expected)


@pytest.mark.parametrize(
    "metadata",
    [
        PagedAttentionMetadata(
            block_tables=((0, 1), (0, 2)),
            num_computed_tokens=(2, 2),
            query_lengths=(1, 1),
            num_readonly_prefix_blocks=(1, 0),
        ),
        PagedAttentionMetadata(
            block_tables=((0, 1, 2), (3, 0, 4)),
            num_computed_tokens=(4, 4),
            query_lengths=(1, 1),
            num_readonly_prefix_blocks=(1, 2),
        ),
    ],
)
def test_paged_attention_rejects_misaligned_or_one_sided_prefix_sharing(
    metadata: PagedAttentionMetadata,
) -> None:
    with pytest.raises(KVCacheError, match="alias a physical block across requests"):
        metadata.validate_block_tables(num_blocks=5, block_size=2)


def test_paged_attention_rejects_readonly_blocks_beyond_the_computed_prefix() -> None:
    metadata = PagedAttentionMetadata(
        block_tables=((0,),),
        num_computed_tokens=(1,),
        query_lengths=(1,),
        num_readonly_prefix_blocks=(1,),
    )

    with pytest.raises(KVCacheError, match="exceed the computed prefix"):
        metadata.validate_block_tables(num_blocks=2, block_size=2)


@pytest.mark.parametrize(
    ("block_tables", "num_computed_tokens"),
    [
        (((0, 0),), (0,)),
        (((0,), (1, 0)), (0, 0)),
    ],
)
def test_paged_attention_rejects_unused_trailing_blocks(
    block_tables: tuple[tuple[int, ...], ...],
    num_computed_tokens: tuple[int, ...],
) -> None:
    cache = _cache()
    batch_size = len(block_tables)
    metadata = PagedAttentionMetadata(
        block_tables=block_tables,
        num_computed_tokens=num_computed_tokens,
        query_lengths=(1,) * batch_size,
    )

    with pytest.raises(KVCacheError, match="unused physical blocks"):
        TorchPagedAttention(cache, metadata).forward(
            "attention",
            torch.randn(batch_size, 1, 2, 4),
            torch.randn(batch_size, 1, 2, 4),
            torch.randn(batch_size, 1, 2, 4),
            scale=0.5,
        )
