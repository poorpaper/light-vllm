from __future__ import annotations

import asyncio
from functools import partial

import pytest
import torch

from light_vllm.modeling.attention.interfaces import AttentionLayerSpec, ModelKVCacheSpec
from light_vllm.modeling.models.interfaces import ForwardBatch
from light_vllm.modeling.models.tiny_attention import (
    TinyAttentionCausalLM,
    TinyAttentionConfig,
)
from light_vllm.runtime.engine import EngineCore
from light_vllm.runtime.execution import (
    DenseAttentionMetadata,
    LocalModelExecutor,
    LocalModelWorker,
    PagedKVCacheConfig,
    TorchDenseAttention,
    TorchPagedAttentionBackend,
)
from light_vllm.runtime.execution.worker import (
    PagedStepHandler,
    StandardDecodeHandler,
)
from light_vllm.runtime.generation import GenerateRequest
from light_vllm.runtime.kv_cache import (
    ContiguousKVCache,
    ContiguousKVCacheConfig,
    ContiguousKVCacheState,
    ContiguousLayerKV,
    FixedKVBlockCapacity,
    KVCacheCapacityError,
    KVCacheError,
    KVCacheNotFoundError,
    PagedKVCacheManager,
    UnboundedKVCacheManager,
)
from light_vllm.runtime.sampling import GreedySampler
from light_vllm.runtime.scheduler import TokenBudgetScheduler


def _updates(*values: float) -> ContiguousKVCacheState:
    tensor = torch.tensor(values, dtype=torch.float32).reshape(1, len(values), 1, 1)
    return ContiguousKVCacheState(layers=(ContiguousLayerKV(keys=tensor, values=tensor + 10),))


def _model_kv_spec(*, num_kv_heads: int = 1, head_size: int = 1) -> ModelKVCacheSpec:
    return ModelKVCacheSpec(
        layers=(
            AttentionLayerSpec(
                layer_id="attention",
                num_query_heads=num_kv_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
            ),
        )
    )


def _tiny_dense_forward(model, token_ids: tuple[int, ...], *, past=None):
    input_ids = torch.tensor([token_ids])
    start = 0 if past is None else past.num_tokens
    positions = torch.arange(start, start + len(token_ids)).unsqueeze(0)
    attention = TorchDenseAttention(
        model.kv_cache_spec,
        DenseAttentionMetadata(
            positions=positions,
            query_lengths=(len(token_ids),),
        ),
        past,
    )
    output = model(
        ForwardBatch(
            input_ids=input_ids,
            positions=positions,
            attention=attention,
        )
    )
    return output, attention


def test_logical_blocks_support_reserve_commit_and_rollback() -> None:
    manager = PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=3, block_size=2))
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
    manager = PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=1, block_size=2))
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
    cache = ContiguousKVCache(_model_kv_spec(), ContiguousKVCacheConfig())
    cache.allocate("request", capacity=2)
    cache.append("request", _updates(1, 2))

    assert cache.cached_tokens("request") == 2
    assert cache.view("request").layers[0].keys.flatten().tolist() == [1, 2]
    with pytest.raises(KVCacheCapacityError):
        cache.append("request", _updates(3))


def test_contiguous_cache_can_truncate_a_rejected_suffix() -> None:
    cache = ContiguousKVCache(_model_kv_spec(), ContiguousKVCacheConfig())
    cache.allocate("request", capacity=3)
    cache.append("request", _updates(1, 2, 3))

    cache.truncate("request", 1)

    assert cache.cached_tokens("request") == 1
    assert cache.view("request").layers[0].keys.flatten().tolist() == [1]
    with pytest.raises(KVCacheError, match="cannot extend"):
        cache.truncate("request", 2)


def test_cache_lease_defers_physical_release_until_execution_finishes() -> None:
    cache = ContiguousKVCache(_model_kv_spec(), ContiguousKVCacheConfig())
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
    full = _tiny_dense_forward(model, (1, 2, 3))[0].logits[:, -1]

    prompt, prompt_attention = _tiny_dense_forward(model, (1, 2))
    cached = _tiny_dense_forward(
        model,
        (3,),
        past=prompt_attention.cache_updates,
    )[0].logits[:, -1]

    torch.testing.assert_close(cached, full)
    assert prompt.logits.shape == (1, 2, 16)


def test_tiny_attention_requires_one_execution_attention_context() -> None:
    model = TinyAttentionCausalLM(
        TinyAttentionConfig(vocab_size=16, hidden_size=8, num_heads=2)
    ).eval()

    with pytest.raises(ValueError, match="requires an attention context"):
        model(ForwardBatch(input_ids=torch.tensor([[1, 2]])))


def test_tiny_attention_delegates_cache_layout_to_attention_context() -> None:
    class RecordingAttention:
        def __init__(self) -> None:
            self.layer_ids: list[str] = []

        def forward(self, layer_id, query, key, value, *, scale):
            self.layer_ids.append(layer_id)
            assert query.shape == key.shape == value.shape == (1, 2, 2, 4)
            assert scale == 0.5
            return query

    model = TinyAttentionCausalLM(
        TinyAttentionConfig(vocab_size=16, hidden_size=8, num_heads=2)
    ).eval()
    attention = RecordingAttention()

    output = model(ForwardBatch(input_ids=torch.tensor([[1, 2]]), attention=attention))

    assert output.logits.shape == (1, 2, 16)
    assert attention.layer_ids == ["attention"]


def test_engine_chunked_prefill_matches_full_sequence_greedy_generation() -> None:
    async def run() -> None:
        torch.manual_seed(11)
        model = TinyAttentionCausalLM(
            TinyAttentionConfig(vocab_size=16, hidden_size=8, num_heads=2)
        ).eval()

        class Forwarder:
            generation = 1
            kv_cache_spec = model.kv_cache_spec
            max_model_tokens = None

            def forward(self, batch: ForwardBatch):
                return model(batch)

        class SessionProvider:
            generation = 1

            def open_session(self):
                return Forwarder()

        request = GenerateRequest(input_ids=(1, 2, 3), max_new_tokens=3)
        expected: list[int] = []
        token_ids = list(request.input_ids)
        for _ in range(request.max_new_tokens):
            logits = _tiny_dense_forward(model, tuple(token_ids))[0].logits
            token_id = int(logits[0, -1].argmax().item())
            expected.append(token_id)
            token_ids.append(token_id)

        cache_config = PagedKVCacheConfig(num_blocks=8, block_size=2)
        logical_cache = PagedKVCacheManager(cache_config)
        worker = LocalModelWorker(
            SessionProvider(),
            partial(
                PagedStepHandler,
                cache_planner=cache_config,
                attention_backend=TorchPagedAttentionBackend(),
            ),
            StandardDecodeHandler(GreedySampler()),
        )
        executor = LocalModelExecutor(worker)
        executor.initialize()
        engine = EngineCore(
            executor,
            TokenBudgetScheduler(
                logical_cache,
                max_num_sequences=2,
                max_num_scheduled_tokens=2,
            ),
        )

        result = await engine.generate(request)
        await engine.close()

        assert result.generated_token_ids == tuple(expected)
        assert logical_cache.num_free_blocks == 8

    asyncio.run(run())
