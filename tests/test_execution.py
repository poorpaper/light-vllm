import pytest
import torch

from light_vllm.modeling.models.interfaces import (
    ForwardBatch,
    KVCacheState,
    LayerKeyValues,
    ModelNotLoadedError,
    ModelOutput,
)
from light_vllm.runtime.execution import (
    ExecutionBatch,
    ExecutionError,
    ExecutionNotReadyError,
    ExecutionRequest,
    LocalModelExecutor,
    LocalModelWorker,
    LocalTokenExecutor,
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
    executor = LocalModelExecutor(LocalModelWorker(IncrementingForwarder(), cache, GreedySampler()))
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
