from __future__ import annotations

import asyncio

import pytest
import torch

from light_vllm.modeling.models.interfaces import (
    ForwardBatch,
    KVCacheState,
    LayerKeyValues,
)
from light_vllm.modeling.models.tiny_attention import (
    TinyAttentionCausalLM,
    TinyAttentionConfig,
)
from light_vllm.runtime.engine import EngineCore
from light_vllm.runtime.execution import LocalModelExecutor
from light_vllm.runtime.generation import GenerateRequest
from light_vllm.runtime.kv_cache import (
    ContiguousKVCache,
    KVCacheCapacityError,
    KVCacheNotFoundError,
    KVCacheSpec,
    PagedKVCacheManager,
    UnboundedKVCacheManager,
)
from light_vllm.runtime.sampling import GreedySampler
from light_vllm.runtime.scheduler import TokenBudgetScheduler


def _updates(*values: float) -> KVCacheState:
    tensor = torch.tensor(values, dtype=torch.float32).reshape(1, len(values), 1, 1)
    return KVCacheState(layers=(LayerKeyValues(keys=tensor, values=tensor + 10),))


def test_logical_blocks_support_reserve_commit_and_rollback() -> None:
    manager = PagedKVCacheManager(num_blocks=3, block_size=2)
    manager.add_request("request")

    first = manager.reserve("request", 3)
    assert first.block_ids == (0, 1)
    assert manager.num_free_blocks == 1

    # 只提交一个 token 后，第二个 block 不再需要，应立即回到空闲池。
    manager.commit("request", 1)
    assert manager.num_free_blocks == 2

    second = manager.reserve("request", 2)
    assert second.block_ids == (0, 1)
    manager.commit("request", 2)
    assert manager.num_free_blocks == 1

    assert manager.free("request")
    assert manager.num_free_blocks == 3


def test_logical_reservation_is_atomic_when_capacity_is_insufficient() -> None:
    manager = PagedKVCacheManager(num_blocks=1, block_size=2)
    manager.add_request("request")

    with pytest.raises(KVCacheCapacityError):
        manager.reserve("request", 3)
    assert manager.num_free_blocks == 1


def test_unbounded_manager_tracks_reservations_without_block_placement() -> None:
    manager = UnboundedKVCacheManager()
    manager.add_request("request")

    first = manager.reserve("request", 3)
    assert first.block_ids is None
    assert first.num_committed_tokens == 0
    assert first.num_reserved_tokens == 3

    manager.commit("request", 1)
    second = manager.reserve("request", 2)
    assert second.block_ids is None
    assert second.num_committed_tokens == 1
    manager.commit("request", 2)

    assert manager.free("request")
    assert not manager.free("request")


def test_contiguous_cache_appends_valid_prefix_and_checks_capacity() -> None:
    cache = ContiguousKVCache(KVCacheSpec(num_layers=1, num_kv_heads=1, head_size=1))
    cache.allocate("request", capacity=2)
    cache.append("request", _updates(1, 2))

    assert cache.cached_tokens("request") == 2
    assert cache.view("request").layers[0].keys.flatten().tolist() == [1, 2]
    with pytest.raises(KVCacheCapacityError):
        cache.append("request", _updates(3))


def test_cache_lease_defers_physical_release_until_execution_finishes() -> None:
    cache = ContiguousKVCache(KVCacheSpec(num_layers=1, num_kv_heads=1, head_size=1))
    cache.allocate("request", capacity=1)
    lease = cache.acquire(("request",))

    assert cache.free("request")
    cache.append("request", _updates(1))
    lease.release()

    with pytest.raises(KVCacheNotFoundError):
        cache.cached_tokens("request")


def test_tiny_attention_cached_logits_match_full_sequence_logits() -> None:
    torch.manual_seed(7)
    model = TinyAttentionCausalLM(
        TinyAttentionConfig(vocab_size=16, hidden_size=8, num_heads=2)
    ).eval()
    full = model(ForwardBatch(input_ids=torch.tensor([[1, 2, 3]]))).logits[:, -1]

    empty = KVCacheState(
        layers=(
            LayerKeyValues(
                keys=torch.empty(1, 0, 2, 4),
                values=torch.empty(1, 0, 2, 4),
            ),
        )
    )
    prompt = model(ForwardBatch(input_ids=torch.tensor([[1, 2]]), kv_cache=empty))
    assert prompt.kv_cache_updates is not None
    cached = model(
        ForwardBatch(input_ids=torch.tensor([[3]]), kv_cache=prompt.kv_cache_updates)
    ).logits[:, -1]

    torch.testing.assert_close(cached, full)


def test_engine_chunked_prefill_matches_full_sequence_greedy_generation() -> None:
    async def run() -> None:
        torch.manual_seed(11)
        model = TinyAttentionCausalLM(
            TinyAttentionConfig(vocab_size=16, hidden_size=8, num_heads=2)
        ).eval()

        class Forwarder:
            generation = 1

            def forward(self, batch: ForwardBatch):
                return model(batch)

        request = GenerateRequest(input_ids=(1, 2, 3), max_new_tokens=3)
        expected: list[int] = []
        token_ids = list(request.input_ids)
        for _ in range(request.max_new_tokens):
            logits = model(ForwardBatch(input_ids=torch.tensor([token_ids]))).logits
            token_id = int(logits[0, -1].argmax().item())
            expected.append(token_id)
            token_ids.append(token_id)

        tensor_cache = ContiguousKVCache(KVCacheSpec(num_layers=1, num_kv_heads=2, head_size=4))
        logical_cache = PagedKVCacheManager(num_blocks=8, block_size=2)
        engine = EngineCore(
            LocalModelExecutor(Forwarder(), tensor_cache, GreedySampler()),
            TokenBudgetScheduler(
                logical_cache,
                max_num_sequences=2,
                max_num_scheduled_tokens=2,
            ),
        )

        result = await engine.generate(request)
        await engine.close()

        assert result.generated_token_ids == tuple(expected)
        assert tensor_cache.num_requests == 0
        assert logical_cache.num_free_blocks == 8

    asyncio.run(run())
