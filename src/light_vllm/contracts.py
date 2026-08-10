from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Everything required to choose and materialize one model."""

    architecture: str
    loader: str = "init"
    model_args: Mapping[str, object] = field(default_factory=dict)
    weights: Path | None = None
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32


@dataclass(frozen=True, slots=True)
class ForwardBatch:
    """The stable input boundary between request preparation and a model."""

    input_ids: Tensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """The minimal output needed by the next runtime layer."""

    logits: Tensor


FinishReason: TypeAlias = Literal["length", "eos"]


class GenerationError(RuntimeError):
    """Base error exposed by the protocol-neutral generation boundary."""


class GenerationNotReadyError(GenerationError):
    """Raised when generation is requested before a model is ready."""


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    """One immutable token generation request."""

    input_ids: tuple[int, ...]
    max_new_tokens: int = 16
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        input_ids = tuple(self.input_ids)
        if not input_ids:
            raise ValueError("input_ids must not be empty")
        if any(type(token_id) is not int or token_id < 0 for token_id in input_ids):
            raise ValueError("input_ids must contain non-negative integers")
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if self.eos_token_id is not None and (
            type(self.eos_token_id) is not int or self.eos_token_id < 0
        ):
            raise ValueError("eos_token_id must be a non-negative integer")
        object.__setattr__(self, "input_ids", input_ids)


@dataclass(frozen=True, slots=True)
class TokenGenerated:
    """One token emitted by an in-progress generation."""

    token_id: int
    position: int


@dataclass(frozen=True, slots=True)
class GenerationFinished:
    """The terminal event of a successful generation."""

    finish_reason: FinishReason


GenerationEvent: TypeAlias = TokenGenerated | GenerationFinished


@dataclass(frozen=True, slots=True)
class GenerateResult:
    """The collected form of a completed generation stream."""

    input_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    finish_reason: FinishReason

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.input_ids + self.generated_token_ids


class ModelFactory(Protocol):
    def __call__(self, spec: ModelSpec) -> nn.Module: ...


class ModelLoader(Protocol):
    def load(self, spec: ModelSpec, factory: ModelFactory) -> nn.Module: ...


class GenerationService(Protocol):
    """Synchronous generation contract used by reference implementations.

    Serving protocols consume ``EngineClient`` instead. Keeping this contract
    synchronous preserves a small offline baseline while the client boundary
    absorbs async and future IPC concerns.
    """

    @property
    def ready(self) -> bool: ...

    def stream(self, request: GenerateRequest) -> Iterator[GenerationEvent]: ...

    def generate(self, request: GenerateRequest) -> GenerateResult: ...
