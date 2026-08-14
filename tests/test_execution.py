import pytest
import torch

from light_vllm.modeling.models.interfaces import (
    ForwardBatch,
    KVCacheState,
    LayerKeyValues,
    ModelNotLoadedError,
    ModelOutput,
)
from light_vllm.modeling.models.tiny_attention import (
    TinyAttentionCausalLM,
    TinyAttentionConfig,
)
from light_vllm.runtime.execution import (
    ContiguousModelWorker,
    ExecutionBatch,
    ExecutionError,
    ExecutionNotReadyError,
    ExecutionRequest,
    LocalModelExecutor,
    LocalTokenExecutor,
    PagedKVCacheConfig,
    PagedModelWorker,
)
from light_vllm.runtime.kv_cache import ContiguousKVCache, KVCacheSpec
from light_vllm.runtime.sampling import GreedySampler


class IncrementingForwarder:
    generation = 1

    def __init__(self, vocab_size: int = 8) -> None:
        self.vocab_size = vocab_size

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        next_ids = (batch.input_ids + 1) % self.vocab_size
        logits = torch.full((*batch.input_ids.shape, self.vocab_size), -1.0)
        logits.scatter_(-1, next_ids.unsqueeze(-1), 1.0)
        updates = KVCacheState(
            layers=(
                LayerKeyValues(
                    keys=batch.input_ids.to(torch.float32).reshape(1, -1, 1, 1),
                    values=batch.input_ids.to(torch.float32).reshape(1, -1, 1, 1),
                ),
            )
        )
        return ModelOutput(logits=logits, kv_cache_updates=updates)


class UnloadedForwarder:
    generation = 0

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        raise ModelNotLoadedError("not loaded")


class InvalidOutputForwarder:
    generation = 1

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        return ModelOutput(logits=torch.zeros(1, 8))


class FixedSampler:
    def sample(self, logits: torch.Tensor) -> tuple[int, ...]:
        return (7,) * logits.shape[0]


class CountingAttentionForwarder:
    generation = 1

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


class ReloadingAttentionForwarder(CountingAttentionForwarder):
    @property
    def kv_cache_spec(self):
        spec = self.model.kv_cache_spec
        self.generation += 1
        return spec


def test_reference_executor_delegates_token_choice_to_sampler() -> None:
    executor = LocalTokenExecutor(IncrementingForwarder(), FixedSampler())

    assert executor.next_token((1, 2)) == 7


def test_reference_executor_exposes_an_unloaded_model_as_not_ready() -> None:
    executor = LocalTokenExecutor(UnloadedForwarder(), GreedySampler())

    assert not executor.ready
    with pytest.raises(ExecutionNotReadyError):
        executor.next_token((1,))


def test_reference_executor_checks_model_output_shape() -> None:
    executor = LocalTokenExecutor(InvalidOutputForwarder(), GreedySampler())

    with pytest.raises(ExecutionError, match="model logits"):
        executor.next_token((1,))


def test_model_executor_handles_prefill_without_sampling_then_decode() -> None:
    cache = ContiguousKVCache(KVCacheSpec(num_layers=1, num_kv_heads=1, head_size=1))
    executor = LocalModelExecutor(
        ContiguousModelWorker(IncrementingForwarder(), cache, GreedySampler())
    )
    executor.add_request("request", capacity=3)

    prefill = executor.execute(
        ExecutionBatch(
            requests=(
                ExecutionRequest(
                    request_id="request",
                    input_token_ids=(1, 2),
                    num_computed_tokens=0,
                    block_ids=None,
                    sampling_required=False,
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
                    num_computed_tokens=2,
                    block_ids=None,
                    sampling_required=True,
                ),
            )
        )
    ).requests[0]

    assert prefill.num_computed_tokens == 2
    assert prefill.token_ids == ()
    assert decode.token_ids == (4,)
    assert cache.cached_tokens("request") == 3


def test_paged_worker_batches_requests_and_matches_full_sequence_attention() -> None:
    torch.manual_seed(23)
    forwarder = CountingAttentionForwarder()
    executor = LocalModelExecutor(
        PagedModelWorker(
            forwarder,
            GreedySampler(),
            PagedKVCacheConfig(num_blocks=8, block_size=2),
        )
    )
    executor.initialize()
    executor.add_request("a", capacity=5)
    executor.add_request("b", capacity=4)

    prefill = executor.execute(
        ExecutionBatch(
            requests=(
                ExecutionRequest("a", (1, 2, 3), 0, (4, 1), True),
                ExecutionRequest("b", (5, 6), 0, (2,), True),
            )
        )
    )

    expected_prefill = tuple(
        int(
            forwarder.model(ForwardBatch(input_ids=torch.tensor([tokens])))
            .logits[0, -1]
            .argmax()
            .item()
        )
        for tokens in ((1, 2, 3), (5, 6))
    )
    assert tuple(result.token_ids[0] for result in prefill.requests) == expected_prefill
    assert forwarder.calls == 1
    torch.testing.assert_close(
        forwarder.position_batches[0],
        torch.tensor([[0, 1, 2], [0, 1, 0]]),
    )

    decode = executor.execute(
        ExecutionBatch(
            requests=(
                ExecutionRequest("a", (expected_prefill[0],), 3, (4, 1), True),
                ExecutionRequest("b", (expected_prefill[1],), 2, (2, 3), True),
            )
        )
    )
    expected_decode = tuple(
        int(
            forwarder.model(ForwardBatch(input_ids=torch.tensor([tokens + (generated,)])))
            .logits[0, -1]
            .argmax()
            .item()
        )
        for tokens, generated in zip(
            ((1, 2, 3), (5, 6)),
            expected_prefill,
            strict=True,
        )
    )

    assert tuple(result.token_ids[0] for result in decode.requests) == expected_decode
    assert forwarder.calls == 2
    torch.testing.assert_close(forwarder.position_batches[1], torch.tensor([[3], [2]]))


def test_paged_worker_rejects_execution_without_block_tables() -> None:
    forwarder = CountingAttentionForwarder()
    executor = LocalModelExecutor(
        PagedModelWorker(
            forwarder,
            GreedySampler(),
            PagedKVCacheConfig(num_blocks=2, block_size=2),
        )
    )
    executor.initialize()
    executor.add_request("request", capacity=2)

    with pytest.raises(ExecutionError, match="requires a block table"):
        executor.execute(
            ExecutionBatch(requests=(ExecutionRequest("request", (1,), 0, None, True),))
        )


def test_paged_worker_rejects_a_model_change_during_cache_initialization() -> None:
    executor = LocalModelExecutor(
        PagedModelWorker(
            ReloadingAttentionForwarder(),
            GreedySampler(),
            PagedKVCacheConfig(num_blocks=2, block_size=2),
        )
    )

    with pytest.raises(ExecutionError, match="model changed while initializing"):
        executor.initialize()
