import asyncio
from functools import partial

import pytest
import torch

from light_vllm.modeling.attention.interfaces import AttentionLayerSpec, ModelKVCacheSpec
from light_vllm.modeling.models.interfaces import (
    ForwardBatch,
    ModelNotLoadedError,
    ModelOutput,
    select_query_states,
)
from light_vllm.modeling.models.tiny_attention import (
    TinyAttentionCausalLM,
    TinyAttentionConfig,
)
from light_vllm.runtime.engine import EngineCore
from light_vllm.runtime.execution import (
    DenseAttentionMetadata,
    DraftTree,
    ExecutionBatch,
    ExecutionCapabilities,
    ExecutionError,
    ExecutionNotReadyError,
    ExecutionOutput,
    ExecutionRequest,
    GreedyTreeAcceptanceSampler,
    LocalModelExecutor,
    LocalModelWorker,
    LocalTokenExecutor,
    ModelStepBatch,
    ModelStepOutput,
    NGramChainProposer,
    PagedKVCacheConfig,
    RequestOutput,
    SpeculativeDecodeHandler,
    TorchDenseAttention,
    TorchPagedAttentionBackend,
)
from light_vllm.runtime.execution.layout import (
    linear_model_step_request,
    linear_query_layout,
)
from light_vllm.runtime.execution.worker import (
    ContiguousStepHandler,
    PagedStepHandler,
    StandardDecodeHandler,
    _validate_positions,
)
from light_vllm.runtime.generation import GenerateRequest
from light_vllm.runtime.kv_cache import (
    ContiguousKVCacheConfig,
    FixedKVBlockCapacity,
    PagedKVCacheManager,
    UnboundedKVCacheManager,
)
from light_vllm.runtime.sampling import GreedySampler
from light_vllm.runtime.scheduler import DecodingBudget, TokenBudgetScheduler


class IncrementingForwarder:
    generation = 1
    max_model_tokens = None
    kv_cache_spec = ModelKVCacheSpec(
        layers=(
            AttentionLayerSpec(
                layer_id="attention",
                num_query_heads=1,
                num_kv_heads=1,
                head_size=1,
            ),
        )
    )

    def __init__(self, vocab_size: int = 8) -> None:
        self.vocab_size = vocab_size

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        next_ids = (batch.input_ids + 1) % self.vocab_size
        logits = torch.full((*batch.input_ids.shape, self.vocab_size), -1.0)
        logits.scatter_(-1, next_ids.unsqueeze(-1), 1.0)
        if batch.attention is not None:
            keys = batch.input_ids.to(torch.float32).reshape(-1, 1, 1)
            batch.attention.forward(
                "attention",
                keys,
                keys,
                keys,
                scale=1.0,
            )
        return ModelOutput(logits=select_query_states(logits, batch))


class UnloadedForwarder:
    generation = 0
    max_model_tokens = None

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        raise ModelNotLoadedError("not loaded")


class InvalidOutputForwarder:
    generation = 1
    kv_cache_spec = None
    max_model_tokens = None

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        return ModelOutput(logits=torch.zeros(2, 8))


class FixedSampler:
    def sample(self, logits: torch.Tensor, metadata=None) -> tuple[int, ...]:
        return (7,) * logits.shape[0]


class CountingAttentionForwarder:
    generation = 1
    max_model_tokens = None

    def __init__(self) -> None:
        self.model = TinyAttentionCausalLM(
            TinyAttentionConfig(vocab_size=16, hidden_size=8, num_heads=2)
        ).eval()
        self.calls = 0
        self.position_batches: list[torch.Tensor] = []

    @property
    def kv_cache_spec(self):
        return self.model.kv_cache_spec

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        self.calls += 1
        assert batch.positions is not None
        self.position_batches.append(batch.positions.clone())
        return self.model(batch)


class StaticSessionProvider:
    def __init__(self, session) -> None:
        self.session = session
        self.generation = session.generation

    def open_session(self):
        return self.session


class UnloadedSessionProvider:
    generation = 0

    def open_session(self):
        raise ModelNotLoadedError("not loaded")


class MutableSessionProvider:
    def __init__(self, session) -> None:
        self.session = session

    @property
    def generation(self) -> int:
        return self.session.generation

    def open_session(self):
        return self.session


class ReloadingDuringForward(CountingAttentionForwarder):
    def __init__(self) -> None:
        super().__init__()
        self.provider: StaticSessionProvider | None = None

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        output = super().forward(batch)
        assert self.provider is not None
        self.provider.generation += 1
        return output


class ReloadingSessionProvider(StaticSessionProvider):
    def open_session(self):
        session = super().open_session()
        self.generation += 1
        return session


def _execution_request(
    request_id: str,
    input_token_ids: tuple[int, ...],
    num_computed_tokens: int,
    block_ids: tuple[int, ...] | None,
    *,
    context_token_ids: tuple[int, ...] | None = None,
    num_lookahead_tokens: int = 0,
    max_output_tokens: int = 1,
) -> ExecutionRequest:
    if context_token_ids is None:
        context_token_ids = (0,) * num_computed_tokens + input_token_ids
    return ExecutionRequest(
        request_id=request_id,
        input_token_ids=input_token_ids,
        context_token_ids=context_token_ids,
        num_computed_tokens=num_computed_tokens,
        num_lookahead_tokens=num_lookahead_tokens,
        max_output_tokens=max_output_tokens,
        block_ids=block_ids,
    )


def test_execution_request_allows_omitted_draft_context() -> None:
    request = ExecutionRequest(
        request_id="request",
        input_token_ids=(3,),
        context_token_ids=None,
        num_computed_tokens=2,
        num_lookahead_tokens=0,
        max_output_tokens=1,
        block_ids=None,
    )

    assert request.context_token_ids is None


@pytest.mark.parametrize(
    ("context_token_ids", "expected_error"),
    [
        ((1, -1, 3), "non-negative integers"),
        ((1, 2, 4), "scheduled slice"),
    ],
)
def test_execution_request_still_validates_present_draft_context(
    context_token_ids: tuple[int, ...],
    expected_error: str,
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        ExecutionRequest(
            request_id="request",
            input_token_ids=(3,),
            context_token_ids=context_token_ids,
            num_computed_tokens=2,
            num_lookahead_tokens=1,
            max_output_tokens=2,
            block_ids=None,
        )


def _contiguous_worker(provider, decode_handler=None) -> LocalModelWorker:
    return LocalModelWorker(
        provider,
        partial(
            ContiguousStepHandler,
            cache_config=ContiguousKVCacheConfig(),
        ),
        decode_handler or StandardDecodeHandler(GreedySampler()),
    )


def _paged_worker(
    provider,
    *,
    num_blocks: int = 8,
    block_size: int = 2,
    decode_handler=None,
) -> LocalModelWorker:
    return LocalModelWorker(
        provider,
        partial(
            PagedStepHandler,
            cache_planner=PagedKVCacheConfig(
                num_blocks=num_blocks,
                block_size=block_size,
            ),
            attention_backend=TorchPagedAttentionBackend(),
        ),
        decode_handler or StandardDecodeHandler(GreedySampler()),
    )


def _dense_tiny_logits(model, token_ids: tuple[int, ...]) -> torch.Tensor:
    input_ids = torch.tensor(token_ids)
    positions = torch.arange(len(token_ids))
    attention = TorchDenseAttention(
        model.kv_cache_spec,
        DenseAttentionMetadata(
            positions=positions,
            query_layouts=(linear_query_layout(len(token_ids)),),
        ),
    )
    return model(
        ForwardBatch(
            input_ids=input_ids,
            positions=positions,
            attention=attention,
        )
    ).logits


def test_step_handler_validates_semantic_positions_before_h2d() -> None:
    _validate_positions((0, 7), max_model_tokens=8)
    _validate_positions((0, 8), max_model_tokens=None)

    for positions in ((-1,), (8,), (True,)):
        with pytest.raises(ExecutionError, match="position"):
            _validate_positions(positions, max_model_tokens=8)


def test_reference_executor_delegates_token_choice_to_sampler() -> None:
    executor = LocalTokenExecutor(StaticSessionProvider(IncrementingForwarder()), FixedSampler())

    assert executor.open_session().next_token((1, 2)) == 7


def test_reference_executor_exposes_an_unloaded_model_as_not_ready() -> None:
    executor = LocalTokenExecutor(UnloadedSessionProvider(), GreedySampler())

    assert not executor.ready
    with pytest.raises(ExecutionNotReadyError):
        executor.open_session()


def test_reference_executor_checks_model_output_shape() -> None:
    executor = LocalTokenExecutor(StaticSessionProvider(InvalidOutputForwarder()), GreedySampler())

    with pytest.raises(ExecutionError, match="model logits"):
        executor.open_session().next_token((1,))


def test_model_executor_handles_prefill_without_sampling_then_decode() -> None:
    executor = LocalModelExecutor(
        _contiguous_worker(StaticSessionProvider(IncrementingForwarder()))
    )
    executor.initialize()
    executor.add_request("request", capacity=3)

    prefill = executor.execute(
        ExecutionBatch(
            requests=(
                ExecutionRequest(
                    request_id="request",
                    input_token_ids=(1, 2),
                    context_token_ids=(1, 2),
                    num_computed_tokens=0,
                    num_lookahead_tokens=0,
                    max_output_tokens=0,
                    block_ids=None,
                ),
            )
        )
    ).requests[0]
    decode = executor.execute(
        ExecutionBatch(
            requests=(
                ExecutionRequest(
                    request_id="request",
                    input_token_ids=(3,),
                    context_token_ids=(1, 2, 3),
                    num_computed_tokens=2,
                    num_lookahead_tokens=0,
                    max_output_tokens=1,
                    block_ids=None,
                ),
            )
        )
    ).requests[0]

    assert prefill.num_input_tokens_computed == 2
    assert prefill.output_token_ids == ()
    assert decode.output_token_ids == (4,)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize(
    ("prompt", "expected_outputs", "expected_queries"),
    [
        (
            (1, 2, 3, 4, 1, 2),
            (3, 4, 5),
            ((1, 2, 3, 4, 1, 2, 3, 4),),
        ),
        (
            (1, 2, 3, 1, 2),
            (3, 4, 5),
            ((1, 2, 3, 1, 2, 3, 1), (4,)),
        ),
        (
            (1, 3, 5),
            (6, 7, 8),
            ((1, 3, 5), (6,), (7,)),
        ),
    ],
)
def test_engine_ngram_speculation_matches_target_generation(
    paged: bool,
    prompt: tuple[int, ...],
    expected_outputs: tuple[int, ...],
    expected_queries: tuple[tuple[int, ...], ...],
) -> None:
    async def run() -> None:
        class RecordingIncrementingForwarder(IncrementingForwarder):
            def __init__(self) -> None:
                super().__init__(vocab_size=16)
                self.queries: list[tuple[int, ...]] = []

            def forward(self, batch: ForwardBatch) -> ModelOutput:
                assert batch.query_start_loc is not None
                self.queries.append(tuple(int(value) for value in batch.input_ids))
                return super().forward(batch)

        forwarder = RecordingIncrementingForwarder()
        provider = StaticSessionProvider(forwarder)
        decode_handler = SpeculativeDecodeHandler(
            NGramChainProposer(min_match_length=2, max_match_length=4),
            GreedySampler(),
            GreedyTreeAcceptanceSampler(),
        )
        if paged:
            worker = _paged_worker(
                provider,
                num_blocks=8,
                block_size=2,
                decode_handler=decode_handler,
            )
            logical_cache = PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=8, block_size=2))
        else:
            worker = _contiguous_worker(provider, decode_handler=decode_handler)
            logical_cache = UnboundedKVCacheManager()
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

        result = await engine.generate(GenerateRequest(input_ids=prompt, max_new_tokens=3))
        await engine.close()

        assert result.generated_token_ids == expected_outputs
        assert forwarder.queries == list(expected_queries)

    asyncio.run(run())


@pytest.mark.parametrize("paged", (False, True), ids=("contiguous", "paged"))
def test_engine_tree_speculation_compacts_a_non_contiguous_root(
    paged: bool,
) -> None:
    async def run() -> None:
        class FixedTreeProposer:
            def propose(self, token_ids: tuple[int, ...], *, max_nodes: int) -> DraftTree:
                tokens = (9, 6)[:max_nodes]
                return DraftTree(tokens, (-1,) * len(tokens))

        class RecordingForwarder(IncrementingForwarder):
            def __init__(self) -> None:
                super().__init__(vocab_size=16)
                self.queries: list[tuple[int, ...]] = []

            def forward(self, batch: ForwardBatch) -> ModelOutput:
                assert batch.query_start_loc is not None
                self.queries.append(tuple(int(value) for value in batch.input_ids))
                return super().forward(batch)

        forwarder = RecordingForwarder()
        provider = StaticSessionProvider(forwarder)
        decode_handler = SpeculativeDecodeHandler(
            FixedTreeProposer(),
            GreedySampler(),
            GreedyTreeAcceptanceSampler(),
        )
        if paged:
            worker = _paged_worker(
                provider,
                num_blocks=8,
                block_size=2,
                decode_handler=decode_handler,
            )
            logical_cache = PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=8, block_size=2))
        else:
            worker = _contiguous_worker(provider, decode_handler=decode_handler)
            logical_cache = UnboundedKVCacheManager()
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

        result = await engine.generate(GenerateRequest(input_ids=(5,), max_new_tokens=3))
        await engine.close()

        assert result.generated_token_ids == (6, 7, 8)
        assert forwarder.queries == [(5, 9, 6), (7,)]

    asyncio.run(run())


def test_paged_step_batches_requests_and_matches_full_sequence_attention() -> None:
    torch.manual_seed(23)
    forwarder = CountingAttentionForwarder()
    executor = LocalModelExecutor(_paged_worker(StaticSessionProvider(forwarder)))
    executor.initialize()
    executor.add_request("a", capacity=5)
    executor.add_request("b", capacity=4)

    prefill = executor.execute(
        ExecutionBatch(
            requests=(
                _execution_request("a", (1, 2, 3), 0, (4, 1)),
                _execution_request("b", (5, 6), 0, (2,)),
            )
        )
    )

    expected_prefill = tuple(
        int(_dense_tiny_logits(forwarder.model, tokens)[-1].argmax().item())
        for tokens in ((1, 2, 3), (5, 6))
    )
    assert tuple(result.output_token_ids[0] for result in prefill.requests) == expected_prefill
    assert forwarder.calls == 1
    torch.testing.assert_close(
        forwarder.position_batches[0],
        torch.tensor([0, 1, 2, 0, 1]),
    )

    decode = executor.execute(
        ExecutionBatch(
            requests=(
                _execution_request("a", (expected_prefill[0],), 3, (4, 1)),
                _execution_request("b", (expected_prefill[1],), 2, (2, 3)),
            )
        )
    )
    expected_decode = tuple(
        int(_dense_tiny_logits(forwarder.model, tokens + (generated,))[-1].argmax().item())
        for tokens, generated in zip(
            ((1, 2, 3), (5, 6)),
            expected_prefill,
            strict=True,
        )
    )

    assert tuple(result.output_token_ids[0] for result in decode.requests) == expected_decode
    assert forwarder.calls == 2
    torch.testing.assert_close(forwarder.position_batches[1], torch.tensor([3, 2]))


def test_paged_step_rejects_execution_without_block_tables() -> None:
    forwarder = CountingAttentionForwarder()
    executor = LocalModelExecutor(_paged_worker(StaticSessionProvider(forwarder), num_blocks=2))
    executor.initialize()
    executor.add_request("request", capacity=2)

    with pytest.raises(ExecutionError, match="requires a block table"):
        executor.execute(ExecutionBatch(requests=(_execution_request("request", (1,), 0, None),)))


def test_worker_rejects_a_model_change_during_step_initialization() -> None:
    executor = LocalModelExecutor(
        _paged_worker(ReloadingSessionProvider(CountingAttentionForwarder()), num_blocks=2)
    )

    with pytest.raises(ExecutionError, match="model changed while initializing"):
        executor.initialize()


def test_worker_finishes_with_its_session_after_runner_reload() -> None:
    forwarder = ReloadingDuringForward()
    provider = StaticSessionProvider(forwarder)
    forwarder.provider = provider
    executor = LocalModelExecutor(_paged_worker(provider, num_blocks=2))
    executor.initialize()
    executor.add_request("request", capacity=2)

    output = executor.execute(
        ExecutionBatch(requests=(_execution_request("request", (1,), 0, (0,)),))
    )

    assert len(output.requests[0].output_token_ids) == 1
    assert not executor.ready
    with pytest.raises(ExecutionError, match="active requests"):
        executor.initialize()
    with pytest.raises(ExecutionNotReadyError):
        executor.add_request("new", capacity=2)


def test_worker_rebuilds_its_step_after_old_requests_finish() -> None:
    old_model = IncrementingForwarder()
    new_model = IncrementingForwarder()
    new_model.generation = 2
    provider = MutableSessionProvider(old_model)
    worker = _contiguous_worker(provider)
    worker.initialize()
    worker.add_request("old", capacity=1)

    provider.session = new_model

    assert not worker.ready
    assert worker.execute(
        ExecutionBatch(requests=(_execution_request("old", (1,), 0, None),))
    ).requests[0].output_token_ids == (2,)
    assert worker.free_request("old")

    worker.initialize()
    worker.add_request("new", capacity=1)

    assert worker.ready
    assert worker.execute(
        ExecutionBatch(requests=(_execution_request("new", (2,), 0, None),))
    ).requests[0].output_token_ids == (3,)


def test_worker_capabilities_come_from_the_selected_step_handler() -> None:
    contiguous = _contiguous_worker(StaticSessionProvider(IncrementingForwarder()))
    paged = _paged_worker(
        StaticSessionProvider(CountingAttentionForwarder()),
        num_blocks=3,
        block_size=2,
    )

    contiguous.initialize()
    paged.initialize()

    assert contiguous.capabilities.max_kv_cache_tokens is None
    assert paged.capabilities.max_kv_cache_tokens == 6


def test_contiguous_step_rejects_a_block_table() -> None:
    executor = LocalModelExecutor(
        _contiguous_worker(StaticSessionProvider(IncrementingForwarder()))
    )
    executor.initialize()
    executor.add_request("request", capacity=2)

    with pytest.raises(ExecutionError, match="does not accept a block table"):
        executor.execute(ExecutionBatch(requests=(_execution_request("request", (1,), 0, (0,)),)))


def test_contiguous_step_can_resume_after_a_speculative_suffix_is_rejected() -> None:
    model = IncrementingForwarder()
    step = ContiguousStepHandler(model, ContiguousKVCacheConfig())
    step.add_request("request", capacity=3)
    first = linear_model_step_request(
        _execution_request("request", (1, 2), 0, None, max_output_tokens=0)
    )
    step.forward(model, ModelStepBatch((first,)))

    step.compact(first, (0,))
    second = linear_model_step_request(
        _execution_request("request", (3,), 1, None, max_output_tokens=0)
    )
    output = step.forward(model, ModelStepBatch((second,)))

    assert output.request_logits(0).shape == (0, model.vocab_size)


def test_model_step_output_keeps_empty_request_slices() -> None:
    logits = torch.zeros((2, 5))
    output = ModelStepOutput(logits, (0, 1, 1, 2))

    assert output.num_requests == 3
    assert output.request_logits(0).shape == (1, 5)
    assert output.request_logits(1).shape == (0, 5)
    assert output.request_logits(2).shape == (1, 5)


def test_standard_decode_samples_the_packed_logits_without_restacking() -> None:
    logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]])
    model_output = ModelStepOutput(logits, (0, 1, 2))

    class Step:
        def forward(self, model, batch):
            return model_output

    class RecordingSampler:
        seen = None

        def sample(self, value, metadata=None):
            self.seen = value
            return (1, 0)

    sampler = RecordingSampler()
    output = StandardDecodeHandler(sampler).execute(
        None,
        ExecutionBatch(
            requests=(
                _execution_request("first", (1,), 0, None),
                _execution_request("second", (2,), 0, None),
            )
        ),
        Step(),
    )

    assert sampler.seen is logits
    assert tuple(request.output_token_ids for request in output.requests) == ((1,), (0,))


@pytest.mark.parametrize(
    "worker",
    [
        _contiguous_worker(StaticSessionProvider(IncrementingForwarder())),
        _paged_worker(
            StaticSessionProvider(CountingAttentionForwarder()),
            num_blocks=2,
        ),
    ],
    ids=("contiguous", "paged"),
)
def test_standard_decode_rejects_lookahead_for_every_step(worker) -> None:
    worker.initialize()
    worker.add_request("request", capacity=2)
    block_ids = None if worker.capabilities.max_kv_cache_tokens is None else (0,)

    with pytest.raises(ExecutionError, match="does not consume lookahead"):
        worker.execute(
            ExecutionBatch(
                requests=(
                    _execution_request(
                        "request",
                        (1,),
                        0,
                        block_ids,
                        num_lookahead_tokens=1,
                        max_output_tokens=2,
                    ),
                )
            )
        )


def test_worker_composes_step_and_decode_handlers_without_mode_branches() -> None:
    class Lease:
        def release(self) -> None:
            return

    class RecordingStep:
        max_kv_cache_tokens = 17

        def __init__(self) -> None:
            self.added: list[tuple[str, int]] = []
            self.freed: list[str] = []

        def add_request(self, request_id: str, *, capacity: int) -> None:
            self.added.append((request_id, capacity))

        def free_request(self, request_id: str) -> None:
            self.freed.append(request_id)

        def acquire(self, request_ids: tuple[str, ...]):
            return Lease()

        def forward(self, model, batch):
            raise AssertionError("decode handler controls when a model step runs")

        def compact(self, request, retained_query_indices: tuple[int, ...]) -> int:
            return 0

    class MultiTokenDecode:
        def __init__(self) -> None:
            self.call = None

        def execute(self, model, batch, step):
            self.call = (model, batch, step)
            return ExecutionOutput(
                requests=(
                    RequestOutput(
                        request_id=batch.requests[0].request_id,
                        num_input_tokens_computed=1,
                        output_token_ids=(2, 3, 4),
                        num_cached_output_tokens=2,
                    ),
                ),
                num_model_tokens_computed=3,
            )

    model = IncrementingForwarder()
    step = RecordingStep()
    decode = MultiTokenDecode()
    worker = LocalModelWorker(
        StaticSessionProvider(model),
        lambda _model: step,
        decode,
    )
    worker.initialize()
    worker.add_request("request", capacity=5)
    batch = ExecutionBatch(
        requests=(
            _execution_request(
                "request",
                (1,),
                0,
                None,
                num_lookahead_tokens=2,
                max_output_tokens=3,
            ),
        )
    )

    output = worker.execute(batch)

    assert worker.capabilities == ExecutionCapabilities(None, 17, model.generation)
    assert step.added == [("request", 5)]
    assert decode.call == (model, batch, step)
    assert output.requests[0].output_token_ids == (2, 3, 4)
    assert output.requests[0].num_cached_output_tokens == 2
    assert worker.free_request("request")
    assert step.freed == ["request"]


def test_local_model_executor_delegates_without_changing_values() -> None:
    class Lease:
        def release(self) -> None:
            return

    class Worker:
        ready = True
        capabilities = ExecutionCapabilities(23, 19)

        def __init__(self) -> None:
            self.calls: list[tuple] = []
            self.lease = Lease()
            self.output = ExecutionOutput(
                requests=(RequestOutput("request", 1, (2,)),),
                num_model_tokens_computed=1,
            )

        def initialize(self) -> None:
            self.calls.append(("initialize",))

        def add_request(self, request_id: str, *, capacity: int) -> None:
            self.calls.append(("add", request_id, capacity))

        def free_request(self, request_id: str) -> bool:
            self.calls.append(("free", request_id))
            return True

        def acquire(self, request_ids: tuple[str, ...]):
            self.calls.append(("acquire", request_ids))
            return self.lease

        def execute(self, batch: ExecutionBatch) -> ExecutionOutput:
            self.calls.append(("execute", batch))
            return self.output

    worker = Worker()
    executor = LocalModelExecutor(worker)
    batch = ExecutionBatch(requests=(_execution_request("request", (1,), 0, None),))

    assert executor.ready
    assert executor.capabilities is worker.capabilities
    executor.initialize()
    executor.add_request("request", capacity=3)
    assert executor.acquire(("request",)) is worker.lease
    assert executor.execute(batch) is worker.output
    assert executor.free_request("request")
    assert worker.calls == [
        ("initialize",),
        ("add", "request", 3),
        ("acquire", ("request",)),
        ("execute", batch),
        ("free", "request"),
    ]


def test_local_model_executor_attaches_completed_step_latency() -> None:
    class Worker:
        ready = True
        capabilities = ExecutionCapabilities(None, None)

        def execute(self, batch: ExecutionBatch) -> ExecutionOutput:
            return ExecutionOutput(
                requests=(RequestOutput("request", 1),),
                num_model_tokens_computed=1,
            )

    class Timer:
        def measure(self, operation):
            return operation(), 0.125

    executor = LocalModelExecutor(Worker(), timer=Timer())
    batch = ExecutionBatch(requests=(_execution_request("request", (1,), 0, None),))

    output = executor.execute(batch)

    assert output.step_elapsed_seconds == pytest.approx(0.125)
    assert output.num_model_tokens_computed == 1
