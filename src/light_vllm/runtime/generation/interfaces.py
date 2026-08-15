from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

FinishReason: TypeAlias = Literal["length", "eos"]


class GenerationError(RuntimeError):
    """生成失败时使用的基础异常。"""


class GenerationNotReadyError(GenerationError):
    """模型还没准备好时抛出。"""


class GenerationRejectedError(GenerationError):
    """请求超出当前引擎明确能力时抛出。"""


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    """一次生成请求，创建后不能修改。"""

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
    """每生成一个 token 就发出一个事件。"""

    token_id: int
    position: int


@dataclass(frozen=True, slots=True)
class GenerationFinished:
    """生成结束时发出的事件。"""

    finish_reason: FinishReason


GenerationEvent: TypeAlias = TokenGenerated | GenerationFinished


@dataclass(frozen=True, slots=True)
class GenerateResult:
    """收集完整生成流后得到的结果。"""

    input_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    finish_reason: FinishReason

    @property
    def token_ids(self) -> tuple[int, ...]:
        return self.input_ids + self.generated_token_ids


class GenerationService(Protocol):
    """同步参考生成接口。"""

    @property
    def ready(self) -> bool: ...

    def stream(self, request: GenerateRequest) -> Iterator[GenerationEvent]: ...

    def generate(self, request: GenerateRequest) -> GenerateResult: ...
