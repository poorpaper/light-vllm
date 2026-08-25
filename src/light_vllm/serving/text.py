"""本地 Hugging Face tokenizer 与增量文本解码 adapter。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from light_vllm.serving.interfaces import (
    ChatMessage,
    IncrementalTextDecoder,
    TextProcessingError,
)


def _token_ids(value: object, *, operation: str) -> tuple[int, ...]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        raise TextProcessingError(f"tokenizer {operation} did not return token IDs")
    token_ids = tuple(value)
    if (
        token_ids
        and isinstance(token_ids[0], Iterable)
        and not isinstance(token_ids[0], (str, bytes))
    ):
        raise TextProcessingError(f"tokenizer {operation} returned a batched result")
    if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
        raise TextProcessingError(f"tokenizer {operation} returned invalid token IDs")
    return token_ids


class _HuggingFaceIncrementalDecoder:
    """按累计 token 解码，并暂存以 replacement character 结尾的片段。"""

    def __init__(self, processor: HuggingFaceTextProcessor) -> None:
        self._processor = processor
        self._token_ids: list[int] = []
        self._prefix_offset = 0
        self._read_offset = 0
        self._finished = False

    def push(self, token_id: int) -> str:
        if self._finished:
            raise TextProcessingError("cannot push tokens after decoder.finish()")
        if type(token_id) is not int or token_id < 0:
            raise TextProcessingError("token_id must be a non-negative integer")
        self._token_ids.append(token_id)
        prefix = self._processor.decode(
            tuple(self._token_ids[self._prefix_offset : self._read_offset])
        )
        text = self._processor.decode(tuple(self._token_ids[self._prefix_offset :]))
        if len(text) <= len(prefix) or text.endswith("\ufffd"):
            return ""
        delta = text[len(prefix) :]
        self._prefix_offset = self._read_offset
        self._read_offset = len(self._token_ids)
        return delta

    def finish(self) -> str:
        if self._finished:
            return ""
        self._finished = True
        prefix = self._processor.decode(
            tuple(self._token_ids[self._prefix_offset : self._read_offset])
        )
        text = self._processor.decode(tuple(self._token_ids[self._prefix_offset :]))
        return text[len(prefix) :] if text.startswith(prefix) else text


class HuggingFaceTextProcessor:
    """只从本地目录加载、禁止远程代码的 tokenizer adapter。"""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None and (type(eos_token_id) is not int or eos_token_id < 0):
            raise TextProcessingError("tokenizer eos_token_id must be a non-negative integer")
        try:
            vocab_size = len(tokenizer)
        except (TypeError, AttributeError) as exc:
            raise TextProcessingError("tokenizer must report its vocabulary size") from exc
        if type(vocab_size) is not int or vocab_size <= 0:
            raise TextProcessingError("tokenizer vocabulary size must be positive")
        self._eos_token_id = eos_token_id
        self._vocab_size = vocab_size

    @classmethod
    def from_pretrained(cls, path: str | Path) -> HuggingFaceTextProcessor:
        tokenizer_path = Path(path)
        if not tokenizer_path.is_dir():
            raise TextProcessingError(f"tokenizer directory does not exist: {tokenizer_path}")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise TextProcessingError(
                "install light-vllm[serve] to use the OpenAI text API"
            ) from exc
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True,
                trust_remote_code=False,
                use_fast=True,
            )
        except Exception as exc:
            raise TextProcessingError(f"cannot load tokenizer from {tokenizer_path}") from exc
        return cls(tokenizer)

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode_prompt(self, prompt: str) -> tuple[int, ...]:
        if not isinstance(prompt, str):
            raise TextProcessingError("prompt must be a string")
        try:
            value = self._tokenizer.encode(prompt, add_special_tokens=False)
        except Exception as exc:
            raise TextProcessingError("cannot tokenize prompt") from exc
        return _token_ids(value, operation="encode")

    def encode_chat(self, messages: tuple[ChatMessage, ...]) -> tuple[int, ...]:
        messages = tuple(messages)
        if not messages:
            raise TextProcessingError("chat messages must not be empty")
        if getattr(self._tokenizer, "chat_template", None) is None:
            raise TextProcessingError("tokenizer does not define a chat template")
        payload = [{"role": item.role, "content": item.content} for item in messages]
        try:
            value = self._tokenizer.apply_chat_template(
                payload,
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception as exc:
            raise TextProcessingError("cannot apply the tokenizer chat template") from exc
        return _token_ids(value, operation="chat template")

    def decode(self, token_ids: tuple[int, ...]) -> str:
        try:
            value = self._tokenizer.decode(
                list(token_ids),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception as exc:
            raise TextProcessingError("cannot decode generated tokens") from exc
        if not isinstance(value, str):
            raise TextProcessingError("tokenizer decode did not return text")
        return value

    def new_decoder(self) -> IncrementalTextDecoder:
        return _HuggingFaceIncrementalDecoder(self)
