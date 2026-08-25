"""OpenAI Chat/Completion wire adapter；模型计算仍由 token 级 EngineClient 完成。"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from math import isfinite
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from light_vllm.runtime.engine.interfaces import EngineClient
from light_vllm.runtime.generation.interfaces import (
    GenerateRequest,
    GenerationError,
    GenerationFinished,
    GenerationNotReadyError,
    GenerationOverloadedError,
    GenerationRejectedError,
    TokenGenerated,
)
from light_vllm.runtime.sampling import SamplingParams
from light_vllm.serving.interfaces import ChatMessage, TextProcessingError, TextProcessor

StrictPositiveInt = Annotated[int, Field(strict=True, ge=1)]
StrictSeed = Annotated[int, Field(strict=True, ge=0, lt=2**63)]
Temperature = Annotated[float, Field(strict=True, ge=0.0, le=2.0, allow_inf_nan=False)]
TopP = Annotated[float, Field(strict=True, gt=0.0, le=1.0, allow_inf_nan=False)]


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_usage: bool = False


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    prompt: str
    max_tokens: StrictPositiveInt = 16
    temperature: Temperature = 1.0
    top_p: TopP = 1.0
    top_k: StrictPositiveInt | None = None
    seed: StrictSeed | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    n: Literal[1] = 1


class ChatMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[ChatMessageRequest] = Field(min_length=1)
    max_tokens: StrictPositiveInt = 16
    temperature: Temperature = 1.0
    top_p: TopP = 1.0
    top_k: StrictPositiveInt | None = None
    seed: StrictSeed | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    n: Literal[1] = 1


@dataclass(frozen=True, slots=True)
class OpenAIServingConfig:
    text_processor: TextProcessor
    served_model_name: str
    request_timeout_seconds: float | None = 300.0
    speculative_decoding: bool = False

    def __post_init__(self) -> None:
        if not self.served_model_name:
            raise ValueError("served_model_name must not be empty")
        timeout = self.request_timeout_seconds
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
            or not isfinite(timeout)
        ):
            raise ValueError("request_timeout_seconds must be positive or None")


@dataclass(frozen=True, slots=True)
class _Usage:
    prompt_tokens: int
    completion_tokens: int

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


@dataclass(frozen=True, slots=True)
class _TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class _TextFinished:
    finish_reason: Literal["stop", "length"]
    usage: _Usage


_TextEvent = _TextDelta | _TextFinished


class _OpenAIRequestError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code


def _error_body(exc: Exception) -> tuple[int, dict[str, object]]:
    if isinstance(exc, _OpenAIRequestError):
        status_code = exc.status_code
        error_type = exc.error_type
        param = exc.param
        code = exc.code
        message = str(exc)
    elif isinstance(exc, GenerationOverloadedError):
        status_code, error_type, param, code = 429, "rate_limit_error", None, "overloaded"
        message = str(exc)
    elif isinstance(exc, GenerationRejectedError):
        status_code, error_type, param, code = 400, "invalid_request_error", None, "capacity"
        message = str(exc)
    elif isinstance(exc, GenerationNotReadyError):
        status_code, error_type, param, code = 503, "server_error", None, "model_not_ready"
        message = "generation service is not ready"
    elif isinstance(exc, TimeoutError):
        status_code, error_type, param, code = 408, "request_timeout", None, "request_timeout"
        message = "generation request timed out"
    elif isinstance(exc, TextProcessingError):
        status_code, error_type, param, code = 400, "invalid_request_error", None, None
        message = str(exc)
    else:
        status_code, error_type, param, code = 500, "server_error", None, None
        message = "generation failed"
    return status_code, {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def openai_error_response(exc: Exception) -> JSONResponse:
    status_code, body = _error_body(exc)
    return JSONResponse(status_code=status_code, content=body)


def _normalize_stop(value: str | list[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    stops = (value,) if isinstance(value, str) else tuple(value)
    if not stops or len(stops) > 4:
        raise _OpenAIRequestError("stop must contain between one and four strings", param="stop")
    if any(not isinstance(stop, str) or not stop for stop in stops):
        raise _OpenAIRequestError("stop strings must not be empty", param="stop")
    return stops


class _StopBuffer:
    def __init__(self, stops: tuple[str, ...]) -> None:
        self._stops = stops
        self._pending = ""
        self._holdback = max(0, max((len(stop) for stop in stops), default=0) - 1)

    def push(self, text: str, *, final: bool = False) -> tuple[str, bool]:
        self._pending += text
        matches = [index for stop in self._stops if (index := self._pending.find(stop)) >= 0]
        if matches:
            index = min(matches)
            visible = self._pending[:index]
            self._pending = ""
            return visible, True
        if final:
            visible, self._pending = self._pending, ""
            return visible, False
        visible_length = max(0, len(self._pending) - self._holdback)
        visible = self._pending[:visible_length]
        self._pending = self._pending[visible_length:]
        return visible, False


async def _close_events(events: AsyncIterator[object]) -> None:
    close = getattr(events, "aclose", None)
    if close is None:
        return
    pending = asyncio.create_task(close())
    try:
        await asyncio.shield(pending)
    except asyncio.CancelledError:
        with suppress(Exception):
            await pending
        raise


async def _consume_text(
    engine_events: AsyncIterator[object],
    *,
    text_processor: TextProcessor,
    prompt_tokens: int,
    stops: tuple[str, ...],
) -> AsyncIterator[_TextEvent]:
    decoder = text_processor.new_decoder()
    stop_buffer = _StopBuffer(stops)
    completion_tokens = 0
    async for event in engine_events:
        if isinstance(event, TokenGenerated):
            completion_tokens += 1
            text, stopped = stop_buffer.push(decoder.push(event.token_id))
            if text:
                yield _TextDelta(text)
            if stopped:
                yield _TextFinished("stop", _Usage(prompt_tokens, completion_tokens))
                return
            continue
        if isinstance(event, GenerationFinished):
            text, stopped = stop_buffer.push(decoder.finish(), final=True)
            if text:
                yield _TextDelta(text)
            finish_reason: Literal["stop", "length"] = (
                "stop" if stopped or event.finish_reason == "eos" else "length"
            )
            yield _TextFinished(finish_reason, _Usage(prompt_tokens, completion_tokens))
            return
        raise GenerationError(f"unsupported generation event: {type(event).__name__}")
    raise GenerationError("generation stream ended without a terminal event")


async def _text_stream(
    engine: EngineClient,
    request: GenerateRequest,
    config: OpenAIServingConfig,
    stops: tuple[str, ...],
) -> AsyncIterator[_TextEvent]:
    events = engine.stream(request)
    try:
        if config.request_timeout_seconds is None:
            async for item in _consume_text(
                events,
                text_processor=config.text_processor,
                prompt_tokens=len(request.input_ids),
                stops=stops,
            ):
                yield item
        else:
            async with asyncio.timeout(config.request_timeout_seconds):
                async for item in _consume_text(
                    events,
                    text_processor=config.text_processor,
                    prompt_tokens=len(request.input_ids),
                    stops=stops,
                ):
                    yield item
    finally:
        await _close_events(events)


def _sampling_request(
    payload: CompletionRequest | ChatCompletionRequest,
    input_ids: tuple[int, ...],
    config: OpenAIServingConfig,
) -> GenerateRequest:
    _validate_model(payload.model, config)
    if not input_ids:
        raise _OpenAIRequestError("prompt must produce at least one token", param="prompt")
    if payload.top_k is not None and payload.top_k > config.text_processor.vocab_size:
        raise _OpenAIRequestError(
            "top_k must not exceed the tokenizer vocabulary size",
            param="top_k",
        )
    sampling = SamplingParams(
        temperature=payload.temperature,
        top_k=payload.top_k,
        top_p=payload.top_p,
        seed=payload.seed,
    )
    if config.speculative_decoding and not sampling.is_greedy:
        raise _OpenAIRequestError(
            "random sampling cannot be combined with speculative decoding in v0.2",
            param="temperature",
        )
    return GenerateRequest(
        input_ids=input_ids,
        max_new_tokens=payload.max_tokens,
        eos_token_id=config.text_processor.eos_token_id,
        sampling=sampling,
    )


def _validate_model(model: str, config: OpenAIServingConfig) -> None:
    if model != config.served_model_name:
        raise _OpenAIRequestError(
            f"model {model!r} is not served",
            status_code=404,
            param="model",
            code="model_not_found",
        )


def _sse_data(value: object) -> bytes:
    return f"data: {json.dumps(value, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()


def _chunk_base(response_id: str, created: int, model: str, object_name: str) -> dict[str, object]:
    return {"id": response_id, "object": object_name, "created": created, "model": model}


async def _completion_sse(
    first: _TextEvent,
    events: AsyncIterator[_TextEvent],
    *,
    response_id: str,
    created: int,
    model: str,
    include_usage: bool,
) -> AsyncIterator[bytes]:
    base = _chunk_base(response_id, created, model, "text_completion")
    try:

        async def emit(item: _TextEvent) -> AsyncIterator[bytes]:
            if isinstance(item, _TextDelta):
                yield _sse_data(
                    base
                    | {
                        "choices": [
                            {"index": 0, "text": item.text, "logprobs": None, "finish_reason": None}
                        ]
                    }
                )
                return
            yield _sse_data(
                base
                | {
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "logprobs": None,
                            "finish_reason": item.finish_reason,
                        }
                    ]
                }
            )
            if include_usage:
                yield _sse_data(base | {"choices": [], "usage": item.usage.as_dict()})

        async for chunk in emit(first):
            yield chunk
        async for item in events:
            async for chunk in emit(item):
                yield chunk
    except Exception as exc:
        _, body = _error_body(exc)
        yield _sse_data(body)
    finally:
        await _close_events(events)
    yield b"data: [DONE]\n\n"


async def _chat_sse(
    first: _TextEvent,
    events: AsyncIterator[_TextEvent],
    *,
    response_id: str,
    created: int,
    model: str,
    include_usage: bool,
) -> AsyncIterator[bytes]:
    base = _chunk_base(response_id, created, model, "chat.completion.chunk")
    try:
        yield _sse_data(
            base
            | {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ]
            }
        )

        async def emit(item: _TextEvent) -> AsyncIterator[bytes]:
            if isinstance(item, _TextDelta):
                yield _sse_data(
                    base
                    | {
                        "choices": [
                            {"index": 0, "delta": {"content": item.text}, "finish_reason": None}
                        ]
                    }
                )
                return
            yield _sse_data(
                base | {"choices": [{"index": 0, "delta": {}, "finish_reason": item.finish_reason}]}
            )
            if include_usage:
                yield _sse_data(base | {"choices": [], "usage": item.usage.as_dict()})

        async for chunk in emit(first):
            yield chunk
        async for item in events:
            async for chunk in emit(item):
                yield chunk
    except Exception as exc:
        _, body = _error_body(exc)
        yield _sse_data(body)
    finally:
        await _close_events(events)
    yield b"data: [DONE]\n\n"


async def _collect_text(events: AsyncIterator[_TextEvent]) -> tuple[str, _TextFinished]:
    text: list[str] = []
    finished: _TextFinished | None = None
    try:
        async for item in events:
            if isinstance(item, _TextDelta):
                text.append(item.text)
            else:
                finished = item
    finally:
        await _close_events(events)
    if finished is None:
        raise GenerationError("text stream ended without a terminal event")
    return "".join(text), finished


def create_openai_router(engine: EngineClient, config: OpenAIServingConfig) -> APIRouter:
    router = APIRouter(prefix="/v1")

    @router.post("/completions", response_model=None)
    async def completions(payload: CompletionRequest):
        try:
            _validate_model(payload.model, config)
            stops = _normalize_stop(payload.stop)
            input_ids = config.text_processor.encode_prompt(payload.prompt)
            request = _sampling_request(payload, input_ids, config)
            events = _text_stream(engine, request, config, stops)
            if payload.stream:
                first = await anext(events)
                response_id = f"cmpl-{uuid4().hex}"
                created = int(time.time())
                return StreamingResponse(
                    _completion_sse(
                        first,
                        events,
                        response_id=response_id,
                        created=created,
                        model=config.served_model_name,
                        include_usage=bool(
                            payload.stream_options and payload.stream_options.include_usage
                        ),
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            text, finished = await _collect_text(events)
            return {
                **_chunk_base(
                    f"cmpl-{uuid4().hex}",
                    int(time.time()),
                    config.served_model_name,
                    "text_completion",
                ),
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "logprobs": None,
                        "finish_reason": finished.finish_reason,
                    }
                ],
                "usage": finished.usage.as_dict(),
            }
        except Exception as exc:
            return openai_error_response(exc)

    @router.post("/chat/completions", response_model=None)
    async def chat_completions(payload: ChatCompletionRequest):
        try:
            _validate_model(payload.model, config)
            stops = _normalize_stop(payload.stop)
            input_ids = config.text_processor.encode_chat(
                tuple(ChatMessage(item.role, item.content) for item in payload.messages)
            )
            request = _sampling_request(payload, input_ids, config)
            events = _text_stream(engine, request, config, stops)
            if payload.stream:
                first = await anext(events)
                response_id = f"chatcmpl-{uuid4().hex}"
                created = int(time.time())
                return StreamingResponse(
                    _chat_sse(
                        first,
                        events,
                        response_id=response_id,
                        created=created,
                        model=config.served_model_name,
                        include_usage=bool(
                            payload.stream_options and payload.stream_options.include_usage
                        ),
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            text, finished = await _collect_text(events)
            return {
                **_chunk_base(
                    f"chatcmpl-{uuid4().hex}",
                    int(time.time()),
                    config.served_model_name,
                    "chat.completion",
                ),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": finished.finish_reason,
                    }
                ],
                "usage": finished.usage.as_dict(),
            }
        except Exception as exc:
            return openai_error_response(exc)

    return router
