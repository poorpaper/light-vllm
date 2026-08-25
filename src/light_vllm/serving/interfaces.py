"""Serving adapter 使用的稳定文本处理契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

ChatRole = Literal["system", "user", "assistant"]


class TextProcessingError(RuntimeError):
    """文本、chat template 或 tokenizer 不能转换时抛出。"""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: ChatRole
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError(f"unsupported chat role: {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("chat message content must be a string")


class IncrementalTextDecoder(Protocol):
    """把逐个 token 转成不会拆坏 Unicode 的文本增量。"""

    def push(self, token_id: int) -> str: ...

    def finish(self) -> str: ...


class TextProcessor(Protocol):
    """OpenAI adapter 所需的 tokenizer 与 chat-template 端口。"""

    @property
    def eos_token_id(self) -> int | None: ...

    @property
    def vocab_size(self) -> int: ...

    def encode_prompt(self, prompt: str) -> tuple[int, ...]: ...

    def encode_chat(self, messages: tuple[ChatMessage, ...]) -> tuple[int, ...]: ...

    def decode(self, token_ids: tuple[int, ...]) -> str: ...

    def new_decoder(self) -> IncrementalTextDecoder: ...
