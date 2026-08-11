import pytest
import torch

from light_vllm.execution import (
    ExecutionError,
    ExecutionNotReadyError,
    GreedyTokenExecutor,
)
from light_vllm.models.api import ForwardBatch, ModelNotLoadedError, ModelOutput


class IncrementingForwarder:
    generation = 1

    def __init__(self, vocab_size: int = 8) -> None:
        self.vocab_size = vocab_size

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        next_ids = (batch.input_ids + 1) % self.vocab_size
        logits = torch.full((*batch.input_ids.shape, self.vocab_size), -1.0)
        logits.scatter_(-1, next_ids.unsqueeze(-1), 1.0)
        return ModelOutput(logits=logits)


class UnloadedForwarder:
    generation = 0

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        raise ModelNotLoadedError("not loaded")


class InvalidOutputForwarder:
    generation = 1

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        return ModelOutput(logits=torch.zeros(1, 8))


def test_local_executor_prepares_input_and_selects_the_best_token() -> None:
    executor = GreedyTokenExecutor(IncrementingForwarder())

    assert executor.ready
    assert executor.next_token((1, 2)) == 3


def test_local_executor_exposes_an_unloaded_model_as_not_ready() -> None:
    executor = GreedyTokenExecutor(UnloadedForwarder())

    assert not executor.ready
    with pytest.raises(ExecutionNotReadyError):
        executor.next_token((1,))


def test_local_executor_checks_model_output_shape() -> None:
    executor = GreedyTokenExecutor(InvalidOutputForwarder())

    with pytest.raises(ExecutionError, match="model logits"):
        executor.next_token((1,))
