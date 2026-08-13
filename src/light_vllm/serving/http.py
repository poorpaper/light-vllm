"""把 ``EngineClient`` 包装成 FastAPI 接口。

这里只处理参数校验、HTTP 状态码、JSON 和 SSE。模型和运行时在
``entrypoints/http.py`` 中创建，测试时可以换成假引擎。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, suppress
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, status
from fastapi.sse import EventSourceResponse, format_sse_event
from pydantic import BaseModel, ConfigDict, Field

from light_vllm.runtime.engine.interfaces import EngineClient
from light_vllm.runtime.generation.interfaces import (
    GenerateRequest,
    GenerateResult,
    GenerationError,
    GenerationEvent,
    GenerationFinished,
    GenerationNotReadyError,
    TokenGenerated,
)

TokenId = Annotated[int, Field(strict=True, ge=0)]
Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]
MAX_PROMPT_TOKENS = 4096


class GenerateHttpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # 这是为了保护参考服务而设置的最大值，不代表模型真正支持的上下文长度。
    # 以后可以由引擎提供更准确的限制。
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
    """在开始返回响应前，把内部异常转换成 HTTP 错误。"""

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
    """把生成事件转换成 SSE 数据。"""

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
    """安全关闭生成流，避免取消和清理同时发生。"""

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
    """把生成事件依次编码成 SSE，并在客户端离开时关闭生成流。

    第一条数据发出后就不能再修改 HTTP 状态码，因此后续错误只能作为
    SSE ``error`` 事件返回。
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
    """用给定的 ``EngineClient`` 创建 FastAPI 应用。

    测试可以传入假引擎；以后换成独立进程引擎也不用修改路由。
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
            # 先读取第一个事件再开始流式响应。这样首次读取失败时还能返回
            # 正确的 HTTP 状态码；开始流式响应后，错误只能写进 SSE。
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
