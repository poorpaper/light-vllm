import pytest
import torch

from light_vllm import (
    ForwardBatch,
    GenerateRequest,
    GenerationError,
    GenerationFinished,
    GenerationNotReadyError,
    GreedyGenerationService,
    ModelOutput,
    TokenGenerated,
)
from light_vllm.runner import ModelNotLoadedError


class IncrementingRunner:
    generation = 1

    def __init__(self, vocab_size: int = 8) -> None:
        self.vocab_size = vocab_size

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        next_ids = (batch.input_ids + 1) % self.vocab_size
        logits = torch.full((*batch.input_ids.shape, self.vocab_size), -1.0)
        logits.scatter_(-1, next_ids.unsqueeze(-1), 1.0)
        return ModelOutput(logits=logits)


class UnloadedRunner:
    generation = 0

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        raise ModelNotLoadedError("not loaded")


class InvalidOutputRunner:
    generation = 1

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        return ModelOutput(logits=torch.zeros(1, 8))


def test_stream_emits_tokens_then_one_terminal_event() -> None:
    service = GreedyGenerationService(IncrementingRunner())

    events = list(service.stream(GenerateRequest(input_ids=(1, 2), max_new_tokens=3)))

    assert events == [
        TokenGenerated(token_id=3, position=0),
        TokenGenerated(token_id=4, position=1),
        TokenGenerated(token_id=5, position=2),
        GenerationFinished(finish_reason="length"),
    ]


def test_generate_collects_the_same_stream() -> None:
    service = GreedyGenerationService(IncrementingRunner())

    result = service.generate(GenerateRequest(input_ids=(1, 2), max_new_tokens=2))

    assert result.input_ids == (1, 2)
    assert result.generated_token_ids == (3, 4)
    assert result.token_ids == (1, 2, 3, 4)
    assert result.finish_reason == "length"


def test_eos_stops_generation_after_emitting_the_token() -> None:
    service = GreedyGenerationService(IncrementingRunner())

    result = service.generate(GenerateRequest(input_ids=(1, 2), max_new_tokens=5, eos_token_id=4))

    assert result.generated_token_ids == (3, 4)
    assert result.finish_reason == "eos"


def test_closed_stream_releases_the_execution_slot() -> None:
    service = GreedyGenerationService(IncrementingRunner())
    first_stream = service.stream(GenerateRequest(input_ids=(1,), max_new_tokens=2))

    assert next(first_stream) == TokenGenerated(token_id=2, position=0)
    first_stream.close()

    assert list(service.stream(GenerateRequest(input_ids=(3,), max_new_tokens=1))) == [
        TokenGenerated(token_id=4, position=0),
        GenerationFinished(finish_reason="length"),
    ]


def test_unloaded_runner_is_exposed_as_not_ready() -> None:
    service = GreedyGenerationService(UnloadedRunner())

    assert not service.ready
    with pytest.raises(GenerationNotReadyError):
        service.generate(GenerateRequest(input_ids=(1,), max_new_tokens=1))


def test_invalid_model_output_is_rejected_at_the_generation_boundary() -> None:
    service = GreedyGenerationService(InvalidOutputRunner())

    with pytest.raises(GenerationError, match="model logits"):
        service.generate(GenerateRequest(input_ids=(1,), max_new_tokens=1))


@pytest.mark.parametrize(
    "changes",
    [
        {"input_ids": ()},
        {"input_ids": (-1,)},
        {"input_ids": (True,)},
        {"max_new_tokens": 0},
        {"eos_token_id": -1},
    ],
)
def test_generate_request_rejects_invalid_values(changes: dict[str, object]) -> None:
    values: dict[str, object] = {"input_ids": (1,), "max_new_tokens": 1}
    values.update(changes)

    with pytest.raises(ValueError):
        GenerateRequest(**values)
