"""FastAPI adapter for the transport-neutral ``EngineClient`` port.

This module owns HTTP concerns only: validation schemas, status codes, JSON,
and SSE encoding. Runtime construction and model loading belong to the HTTP
entrypoint, which makes this adapter easy to test with a stub engine.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, suppress
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, status
from fastapi.sse import EventSourceResponse, format_sse_event
from pydantic import BaseModel, ConfigDict, Field

from light_vllm.contracts import (
    GenerateRequest,
    GenerateResult,
    GenerationError,
    GenerationEvent,
    GenerationFinished,
    GenerationNotReadyError,
    TokenGenerated,
)
from light_vllm.engine import EngineClient

TokenId = Annotated[int, Field(strict=True, ge=0)]
Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]
MAX_PROMPT_TOKENS = 4096


class GenerateHttpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # This is an adapter-level safety ceiling for the reference server, not a
    # model context-length contract. Future engines may expose a configured
    # limit while the transport-neutral GenerateRequest stays reusable.
    input_ids: list[TokenId] = Field(min_length=1, max_length=MAX_PROMPT_TOKENS)
    max_new_tokens: Annotated[int, Field(strict=True, ge=1, le=4096)] = 16
    eos_token_id: TokenId | None = None

    def to_contract(self) -> GenerateRequest:
        return GenerateRequest(
            input_ids=tuple(self.input_ids),
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.eos_token_id,
        )


class GenerateHttpResponse(BaseModel):
    input_ids: list[int]
    generated_token_ids: list[int]
    token_ids: list[int]
    finish_reason: Literal["length", "eos"]

    @classmethod
    def from_contract(cls, result: GenerateResult) -> GenerateHttpResponse:
        return cls(
            input_ids=list(result.input_ids),
            generated_token_ids=list(result.generated_token_ids),
            token_ids=list(result.token_ids),
            finish_reason=result.finish_reason,
        )


class StatusResponse(BaseModel):
    status: Literal["ok", "ready"]


class TokenEventData(BaseModel):
    token_id: int
    position: int


class FinishedEventData(BaseModel):
    finish_reason: Literal["length", "eos"]


class ErrorEventData(BaseModel):
    code: Literal["not_ready", "generation_failed"]
    detail: str


def _http_error(exc: Exception) -> HTTPException:
    """Map domain failures before an HTTP response has started."""

    if isinstance(exc, GenerationNotReadyError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="generation service is not ready",
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="generation failed",
    )


def _encode_event(event: GenerationEvent) -> bytes:
    """Translate one domain event into its HTTP/SSE wire representation."""

    if isinstance(event, TokenGenerated):
        data = TokenEventData(token_id=event.token_id, position=event.position)
        return format_sse_event(
            event="token",
            id=str(event.position),
            data_str=data.model_dump_json(),
        )
    if isinstance(event, GenerationFinished):
        data = FinishedEventData(finish_reason=event.finish_reason)
        return format_sse_event(
            event="done",
            data_str=data.model_dump_json(),
        )
    raise TypeError(f"unsupported generation event: {type(event).__name__}")


def _stream_error_event(exc: Exception) -> bytes:
    if isinstance(exc, GenerationNotReadyError):
        data = ErrorEventData(code="not_ready", detail="generation service is not ready")
    else:
        data = ErrorEventData(code="generation_failed", detail="generation failed")
    return format_sse_event(event="error", data_str=data.model_dump_json())


async def _close_engine_stream(events: AsyncIterator[GenerationEvent]) -> None:
    """Close an engine stream without racing cancellation against cleanup."""

    aclose = getattr(events, "aclose", None)
    if aclose is not None:
        pending = asyncio.create_task(aclose())
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError:
            with suppress(Exception):
                await pending
            raise
        return

    close = getattr(events, "close", None)
    if close is not None:
        close()


async def _encoded_stream(
    first_event: GenerationEvent, events: AsyncIterator[GenerationEvent]
) -> AsyncIterator[bytes]:
    """Encode an engine stream and always close it when the consumer leaves.

    HTTP status and headers are already committed after the first body event,
    so later failures must become an SSE ``error`` event instead of an HTTP
    error response.
    """

    finished = False
    try:
        yield _encode_event(first_event)
        finished = isinstance(first_event, GenerationFinished)

        async for event in events:
            if finished:
                raise GenerationError("generation stream emitted an event after completion")
            yield _encode_event(event)
            finished = isinstance(event, GenerationFinished)

        if not finished:
            raise GenerationError("generation stream ended without a terminal event")
    except Exception as exc:
        yield _stream_error_event(exc)
    finally:
        await _close_engine_stream(events)


def create_http_app(
    engine: EngineClient,
    *,
    lifespan: Lifespan | None = None,
) -> FastAPI:
    """Create a reusable HTTP application around a pre-built engine client.

    Keeping construction outside this factory lets tests inject a stub and lets
    future entrypoints choose an in-process or IPC client without changing any
    route.
    """

    app = FastAPI(title="light-vllm", lifespan=lifespan)

    @app.get("/healthz", response_model=StatusResponse)
    def health() -> StatusResponse:
        return StatusResponse(status="ok")

    @app.get(
        "/readyz",
        response_model=StatusResponse,
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Model is not ready"}},
    )
    def readiness() -> StatusResponse:
        if not engine.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="generation service is not ready",
            )
        return StatusResponse(status="ready")

    @app.post(
        "/generate",
        response_model=GenerateHttpResponse,
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Model is not ready"}},
    )
    async def generate(payload: GenerateHttpRequest) -> GenerateHttpResponse:
        try:
            result = await engine.generate(payload.to_contract())
        except Exception as exc:
            raise _http_error(exc) from exc
        return GenerateHttpResponse.from_contract(result)

    @app.post(
        "/generate/stream",
        response_model=None,
        responses={
            status.HTTP_200_OK: {"content": {"text/event-stream": {}}},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Model is not ready"},
        },
    )
    async def stream(payload: GenerateHttpRequest) -> EventSourceResponse:
        events = engine.stream(payload.to_contract())
        try:
            # Pull once before creating the streaming response. Failures at
            # this point can still be represented by an honest HTTP status;
            # subsequent failures must be encoded inside the SSE stream.
            first_event = await anext(events)
        except StopAsyncIteration as exc:
            with suppress(Exception):
                await _close_engine_stream(events)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="generation stream ended without an event",
            ) from exc
        except asyncio.CancelledError:
            await _close_engine_stream(events)
            raise
        except Exception as exc:
            with suppress(Exception):
                await _close_engine_stream(events)
            raise _http_error(exc) from exc

        return EventSourceResponse(
            _encoded_stream(first_event, events),
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
