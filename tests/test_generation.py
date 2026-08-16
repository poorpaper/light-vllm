import pytest

from light_vllm import (
    GenerateRequest,
    GenerationError,
    GenerationFinished,
    GenerationNotReadyError,
    ReferenceGenerationService,
    TokenGenerated,
)
from light_vllm.runtime.execution import ExecutionError, ExecutionNotReadyError


class _SelfSessionExecutor:
    def open_session(self):
        return self


class IncrementingExecutor(_SelfSessionExecutor):
    ready = True

    def __init__(self, vocab_size: int = 8) -> None:
        self.vocab_size = vocab_size

    def next_token(self, token_ids: tuple[int, ...]) -> int:
        return (token_ids[-1] + 1) % self.vocab_size


class UnreadyExecutor(_SelfSessionExecutor):
    ready = False

    def next_token(self, token_ids: tuple[int, ...]) -> int:
        raise ExecutionNotReadyError("not ready")


class FailingExecutor(_SelfSessionExecutor):
    ready = True

    def next_token(self, token_ids: tuple[int, ...]) -> int:
        raise ExecutionError("model logits have an invalid shape")


def test_stream_emits_tokens_then_one_terminal_event() -> None:
    service = ReferenceGenerationService(IncrementingExecutor())

    events = list(service.stream(GenerateRequest(input_ids=(1, 2), max_new_tokens=3)))

    assert events == [
        TokenGenerated(token_id=3, position=0),
        TokenGenerated(token_id=4, position=1),
        TokenGenerated(token_id=5, position=2),
        GenerationFinished(finish_reason="length"),
    ]


def test_generate_collects_the_same_stream() -> None:
    service = ReferenceGenerationService(IncrementingExecutor())

    result = service.generate(GenerateRequest(input_ids=(1, 2), max_new_tokens=2))

    assert result.input_ids == (1, 2)
    assert result.generated_token_ids == (3, 4)
    assert result.token_ids == (1, 2, 3, 4)
    assert result.finish_reason == "length"


def test_eos_stops_generation_after_emitting_the_token() -> None:
    service = ReferenceGenerationService(IncrementingExecutor())

    result = service.generate(GenerateRequest(input_ids=(1, 2), max_new_tokens=5, eos_token_id=4))

    assert result.generated_token_ids == (3, 4)
    assert result.finish_reason == "eos"


def test_closed_stream_releases_the_execution_slot() -> None:
    service = ReferenceGenerationService(IncrementingExecutor())
    first_stream = service.stream(GenerateRequest(input_ids=(1,), max_new_tokens=2))

    assert next(first_stream) == TokenGenerated(token_id=2, position=0)
    first_stream.close()

    assert list(service.stream(GenerateRequest(input_ids=(3,), max_new_tokens=1))) == [
        TokenGenerated(token_id=4, position=0),
        GenerationFinished(finish_reason="length"),
    ]


def test_unready_executor_is_exposed_as_not_ready() -> None:
    service = ReferenceGenerationService(UnreadyExecutor())

    assert not service.ready
    with pytest.raises(GenerationNotReadyError):
        service.generate(GenerateRequest(input_ids=(1,), max_new_tokens=1))


def test_execution_failure_is_exposed_as_generation_error() -> None:
    service = ReferenceGenerationService(FailingExecutor())

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
