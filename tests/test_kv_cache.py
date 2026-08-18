from __future__ import annotations

import asyncio
from functools import partial

import pytest
import torch

from light_vllm.modeling.attention.interfaces import AttentionLayerSpec, ModelKVCacheSpec
from light_vllm.modeling.models.interfaces import ForwardBatch, ModelOutput
from light_vllm.modeling.models.tiny_attention import (
    TinyAttentionCausalLM,
    TinyAttentionConfig,
)
from light_vllm.runtime.engine import EngineCore
from light_vllm.runtime.execution import (
    DenseAttentionMetadata,
    GreedyAcceptanceSampler,
    LocalModelExecutor,
    LocalModelWorker,
    NGramSpeculativeDecodeHandler,
    NGramTokenProposer,
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
from light_vllm.runtime.scheduler import DecodingBudget, TokenBudgetScheduler


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


def _add_logical_request(
    manager,
    request_id: str,
    token_ids: tuple[int, ...],
    *,
    cache_epoch: int = 1,
    max_num_committed_tokens: int | None = None,
):
    """测试辅助：准入一个确定能放下的逻辑 KV 请求。"""

    admission = manager.try_add_request(
        request_id,
        token_ids=token_ids,
        max_num_committed_tokens=max_num_committed_tokens or len(token_ids),
        cache_epoch=cache_epoch,
    )
    assert admission is not None
    return admission


def test_logical_blocks_support_reserve_commit_and_rollback() -> None:
    manager = PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=3, block_size=2))
    _add_logical_request(manager, "request", (1, 2, 3))

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


def test_logical_admission_is_atomic_when_completion_capacity_is_insufficient() -> None:
    manager = PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=1, block_size=2))
    admission = manager.try_add_request(
        "request",
        token_ids=(1, 2, 3),
        max_num_committed_tokens=3,
        cache_epoch=1,
    )

    assert admission is None
    assert manager.num_free_blocks == 1
    assert manager.stats.claimed_token_slots == 0


def test_completion_claim_keeps_an_admitted_request_progressing() -> None:
    manager = PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=6, block_size=1))
    _add_logical_request(
        manager,
        "running",
        (1,),
        max_num_committed_tokens=4,
    )

    blocked = manager.try_add_request(
        "waiting",
        token_ids=(2,),
        max_num_committed_tokens=3,
        cache_epoch=1,
    )

    assert blocked is None
    assert manager.stats.claimed_token_slots == 4
    reservation = manager.reserve("running", 4)
    assert reservation.block_ids == (0, 1, 2, 3)
    manager.commit("running", 4)
    assert manager.stats.used_token_slots == 4
    assert manager.stats.claimed_token_slots == 0

    manager.free("running")
    assert (
        manager.try_add_request(
            "waiting",
            token_ids=(2,),
            max_num_committed_tokens=3,
            cache_epoch=1,
        )
        is not None
    )


def test_partial_commit_turns_unused_blocks_back_into_completion_claims() -> None:
    manager = PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=3, block_size=1))
    _add_logical_request(
        manager,
        "request",
        (1,),
        max_num_committed_tokens=3,
    )
    assert manager.stats.claimed_token_slots == 3

    manager.reserve("request", 3)
    assert manager.stats.used_token_slots == 3
    assert manager.stats.claimed_token_slots == 0

    manager.commit("request", 1)
    assert manager.stats.used_token_slots == 1
    assert manager.stats.claimed_token_slots == 2
    assert manager.stats.used_token_slots + manager.stats.claimed_token_slots == 3

    manager.free("request")
    assert manager.stats.used_token_slots == 0
    assert manager.stats.claimed_token_slots == 0


def test_best_effort_reservation_keeps_its_kv_watermark() -> None:
    manager = PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=4, block_size=1))
    _add_logical_request(
        manager,
        "guarded",
        (1,),
        max_num_committed_tokens=2,
    )
    admitted = manager.try_add_request(
        "best-effort",
        token_ids=(2,),
        max_num_committed_tokens=4,
        cache_epoch=1,
        min_free_token_slots=1,
        guarantee_completion=False,
    )
    assert admitted is not None
    assert manager.stats.claimed_token_slots == 2

    manager.reserve("best-effort", 1)
    manager.commit("best-effort", 1)
    with pytest.raises(KVCacheCapacityError, match="watermark"):
        manager.reserve("best-effort", 1)

    manager.free("guarded")
    reservation = manager.reserve("best-effort", 1)
    assert reservation.block_ids == (0, 1)


def test_unbounded_manager_tracks_reservations_without_block_placement() -> None:
    manager = UnboundedKVCacheManager()
    _add_logical_request(manager, "request", (1, 2, 3))

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


def test_prefix_cache_reuses_only_committed_full_prompt_blocks() -> None:
    manager = PagedKVCacheManager(
        FixedKVBlockCapacity(num_blocks=4, block_size=2),
        enable_prefix_caching=True,
    )
    tokens = (1, 2, 3, 4, 5)
    assert _add_logical_request(manager, "warm", tokens).num_cached_tokens == 0
    warm = manager.reserve("warm", len(tokens))
    manager.commit("warm", len(tokens))
    assert warm.block_ids == (0, 1, 2)
    assert manager.free("warm")

    match = _add_logical_request(manager, "hit", (1, 2, 3, 4, 9))
    assert match.num_cached_tokens == 4
    reservation = manager.reserve("hit", 1)
    assert reservation.block_ids[:2] == (0, 1)
    assert reservation.num_readonly_prefix_blocks == 2
    assert manager.free("hit")


def test_prefix_cache_leaves_the_last_full_prompt_block_for_logits() -> None:
    manager = PagedKVCacheManager(
        FixedKVBlockCapacity(num_blocks=3, block_size=2),
        enable_prefix_caching=True,
    )
    tokens = (1, 2, 3, 4)
    _add_logical_request(manager, "warm", tokens)
    manager.reserve("warm", len(tokens))
    manager.commit("warm", len(tokens))
    manager.free("warm")

    match = _add_logical_request(manager, "hit", tokens)
    assert match.num_cached_tokens == 2
    manager.free("hit")


def test_prefix_preview_does_not_pin_or_allocate_blocks() -> None:
    manager = PagedKVCacheManager(
        FixedKVBlockCapacity(num_blocks=3, block_size=2),
        enable_prefix_caching=True,
    )
    tokens = (1, 2, 3)
    _add_logical_request(manager, "warm", tokens)
    manager.reserve("warm", len(tokens))
    manager.commit("warm", len(tokens))
    manager.free("warm")
    before = manager.stats

    match = manager.preview_prefix(token_ids=tokens, cache_epoch=1)

    assert match.num_cached_tokens == 2
    assert manager.stats == before
    assert not manager.free("preview")


def test_prefix_cache_does_not_publish_reserved_or_partial_blocks() -> None:
    manager = PagedKVCacheManager(
        FixedKVBlockCapacity(num_blocks=3, block_size=2),
        enable_prefix_caching=True,
    )
    tokens = (1, 2, 3)
    _add_logical_request(manager, "partial", tokens)
    manager.reserve("partial", len(tokens))
    manager.commit("partial", 1)
    manager.free("partial")

    match = _add_logical_request(manager, "miss", tokens)
    assert match.num_cached_tokens == 0
    manager.free("miss")


def test_prefix_cache_is_invalidated_when_the_worker_epoch_changes() -> None:
    manager = PagedKVCacheManager(
        FixedKVBlockCapacity(num_blocks=2, block_size=2),
        enable_prefix_caching=True,
    )
    tokens = (1, 2, 3)
    _add_logical_request(manager, "old", tokens)
    manager.reserve("old", len(tokens))
    manager.commit("old", len(tokens))
    manager.free("old")

    match = _add_logical_request(manager, "new", tokens, cache_epoch=2)
    assert match.num_cached_tokens == 0
    assert manager.num_free_blocks == 2
    manager.free("new")


def test_prefix_cache_evicts_the_least_recently_used_unreferenced_block() -> None:
    manager = PagedKVCacheManager(
        FixedKVBlockCapacity(num_blocks=3, block_size=1),
        enable_prefix_caching=True,
    )
    for request_id, tokens in (("a", (1, 9)), ("b", (2, 9))):
        _add_logical_request(manager, request_id, tokens)
        manager.reserve(request_id, len(tokens))
        manager.commit(request_id, len(tokens))
        manager.free(request_id)

    assert _add_logical_request(manager, "touch-a", (1, 8)).num_cached_tokens == 1
    manager.free("touch-a")
    _add_logical_request(manager, "other", (7, 9))
    manager.reserve("other", 2)
    manager.commit("other", 2)
    manager.free("other")

    assert _add_logical_request(manager, "miss-b", (2, 8)).num_cached_tokens == 0
    manager.free("miss-b")
    assert _add_logical_request(manager, "hit-a", (1, 8)).num_cached_tokens == 1
    manager.free("hit-a")


def test_prefix_cache_never_evicts_a_block_used_by_an_active_request() -> None:
    manager = PagedKVCacheManager(
        FixedKVBlockCapacity(num_blocks=2, block_size=1),
        enable_prefix_caching=True,
    )
    _add_logical_request(manager, "warm", (1, 9))
    manager.reserve("warm", 2)
    manager.commit("warm", 2)
    manager.free("warm")

    assert _add_logical_request(manager, "hit", (1, 8)).num_cached_tokens == 1
    blocked = manager.try_add_request(
        "blocked",
        token_ids=(2, 9),
        max_num_committed_tokens=2,
        cache_epoch=1,
    )
    assert blocked is None

    reservation = manager.reserve("hit", 1)
    assert reservation.block_ids == (0, 1)
    manager.commit("hit", 1)
    manager.free("hit")


def test_prefix_cache_evicts_a_leaf_before_its_reachable_parent() -> None:
    manager = PagedKVCacheManager(
        FixedKVBlockCapacity(num_blocks=4, block_size=1),
        enable_prefix_caching=True,
    )
    warm_tokens = (1, 2, 3, 9)
    _add_logical_request(manager, "warm", warm_tokens)
    manager.reserve("warm", len(warm_tokens))
    manager.commit("warm", len(warm_tokens))
    manager.free("warm")

    for request_id, token_id in (("first", 7), ("second", 8)):
        _add_logical_request(manager, request_id, (token_id,))
        manager.reserve(request_id, 1)
        manager.commit(request_id, 1)
    manager.free("first")
    manager.free("second")

    hit = _add_logical_request(manager, "hit", (1, 2, 6))
    assert hit.num_cached_tokens == 2


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


def test_engine_reuses_a_shared_prompt_prefix_without_changing_generation() -> None:
    async def run() -> None:
        torch.manual_seed(31)
        model = TinyAttentionCausalLM(
            TinyAttentionConfig(vocab_size=16, hidden_size=8, num_heads=2)
        ).eval()

        class Forwarder:
            generation = 1
            kv_cache_spec = model.kv_cache_spec
            max_model_tokens = None

            def __init__(self) -> None:
                self.query_lengths: list[tuple[int, ...]] = []

            def forward(self, batch: ForwardBatch):
                lengths = batch.sequence_lengths or (batch.input_ids.shape[1],)
                self.query_lengths.append(lengths)
                return model(batch)

        forwarder = Forwarder()

        class SessionProvider:
            generation = 1

            def open_session(self):
                return forwarder

        cache_config = PagedKVCacheConfig(num_blocks=8, block_size=2)
        logical_cache = PagedKVCacheManager(
            cache_config,
            enable_prefix_caching=True,
        )
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
                max_num_sequences=1,
                max_num_scheduled_tokens=8,
            ),
        )

        await engine.generate(GenerateRequest(input_ids=(1, 2, 3, 4, 5), max_new_tokens=1))
        second_prompt = (1, 2, 3, 4, 6)
        result = await engine.generate(GenerateRequest(input_ids=second_prompt, max_new_tokens=1))
        expected = int(_tiny_dense_forward(model, second_prompt)[0].logits[0, -1].argmax())
        await engine.close()

        assert result.generated_token_ids == (expected,)
        assert forwarder.query_lengths == [(5,), (1,)]
        assert logical_cache.num_free_blocks == 8

    asyncio.run(run())


def test_engine_speculates_after_reusing_a_shared_prompt_prefix() -> None:
    async def run() -> None:
        class Forwarder:
            generation = 1
            max_model_tokens = None
            kv_cache_spec = _model_kv_spec()

            def __init__(self) -> None:
                self.queries: list[tuple[int, ...]] = []

            def forward(self, batch: ForwardBatch) -> ModelOutput:
                query_length = (batch.sequence_lengths or (batch.input_ids.shape[1],))[0]
                self.queries.append(
                    tuple(int(value) for value in batch.input_ids[0, :query_length])
                )
                next_ids = (batch.input_ids + 1) % 16
                logits = torch.full((*batch.input_ids.shape, 16), -1.0)
                logits.scatter_(-1, next_ids.unsqueeze(-1), 1.0)
                keys = batch.input_ids.to(torch.float32).reshape(1, -1, 1, 1)
                assert batch.attention is not None
                batch.attention.forward("attention", keys, keys, keys, scale=1.0)
                return ModelOutput(logits=logits)

        forwarder = Forwarder()

        class SessionProvider:
            generation = 1

            def open_session(self):
                return forwarder

        cache_config = PagedKVCacheConfig(num_blocks=8, block_size=2)
        logical_cache = PagedKVCacheManager(cache_config, enable_prefix_caching=True)
        worker = LocalModelWorker(
            SessionProvider(),
            partial(
                PagedStepHandler,
                cache_planner=cache_config,
                attention_backend=TorchPagedAttentionBackend(),
            ),
            NGramSpeculativeDecodeHandler(
                NGramTokenProposer(min_match_length=2, max_match_length=4),
                GreedySampler(),
                GreedyAcceptanceSampler(),
            ),
        )
        executor = LocalModelExecutor(worker)
        executor.initialize()
        engine = EngineCore(
            executor,
            TokenBudgetScheduler(
                logical_cache,
                max_num_sequences=1,
                max_num_scheduled_tokens=8,
                decoding_budget=DecodingBudget(
                    num_lookahead_tokens=2,
                    max_output_tokens=3,
                ),
            ),
        )
        prompt = (1, 2, 3, 4, 1, 2)

        first = await engine.generate(GenerateRequest(input_ids=prompt, max_new_tokens=3))
        second = await engine.generate(GenerateRequest(input_ids=prompt, max_new_tokens=3))
        await engine.close()

        assert first.generated_token_ids == second.generated_token_ids == (3, 4, 5)
        assert forwarder.queries == [
            (1, 2, 3, 4, 1, 2, 3, 4),
            (1, 2, 3, 4),
        ]
        assert logical_cache.num_free_blocks == 8

    asyncio.run(run())
