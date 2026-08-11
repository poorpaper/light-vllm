"""把同步生成服务接到异步 Engine 接口。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import suppress
from typing import TypeVar

from light_vllm.generation.api import (
    GenerateRequest,
    GenerateResult,
    GenerationError,
    GenerationEvent,
    GenerationFinished,
    GenerationService,
    TokenGenerated,
)


class _StreamEnded(Exception):
    """表示同步迭代器已经正常结束。"""


_ResultT = TypeVar("_ResultT")


def _next_event(events: Iterator[GenerationEvent]) -> GenerationEvent:
    # StopIteration 不能直接穿过 asyncio Future，所以先换成普通异常，
    # 再回到事件循环处理。
    try:
        return next(events)
    except StopIteration as exc:
        raise _StreamEnded from exc


async def _run_sync_step(
    worker: Executor,
    callback: Callable[[], _ResultT],
) -> _ResultT:
    """在线程中执行一步同步操作；收到取消时，先等这一步结束。"""

    loop = asyncio.get_running_loop()
    pending = loop.run_in_executor(worker, callback)
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        # Python 不能强行停止线程。等当前操作结束后再清理，避免
        # ``next()`` 和 ``close()`` 同时操作同一个迭代器。
        with suppress(Exception):
            await pending
        raise


class InProcessEngineClient:
    """在当前进程内，把同步生成服务包装成异步接口。"""

    def __init__(self, service: GenerationService) -> None:
        self._service = service
        # 让并发请求先在事件循环中排队，等待时不占用工作线程。
        self._admission_lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self._service.ready

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        """读取完整事件流，并整理成生成结果。"""

        generated_token_ids: list[int] = []
        finish_reason = None
        events = self.stream(request)

        try:
            async for event in events:
                if finish_reason is not None:
                    raise GenerationError("generation stream emitted an event after completion")
                if isinstance(event, TokenGenerated):
                    generated_token_ids.append(event.token_id)
                elif isinstance(event, GenerationFinished):
                    finish_reason = event.finish_reason
        finally:
            await events.aclose()

        if finish_reason is None:
            raise GenerationError("generation stream ended without a terminal event")

        return GenerateResult(
            input_ids=request.input_ids,
            generated_token_ids=tuple(generated_token_ids),
            finish_reason=finish_reason,
        )

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[GenerationEvent, None]:
        """把同步事件流包装成支持取消的异步事件流。"""

        async with self._admission_lock:
            # 每个正在执行的请求使用一个专用单线程。创建、读取和关闭事件流
            # 都在这个线程里完成。
            worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="light-vllm-infer")
            try:
                events = await _run_sync_step(worker, lambda: self._service.stream(request))
                try:
                    while True:
                        try:
                            yield await _run_sync_step(worker, lambda: _next_event(events))
                        except _StreamEnded:
                            return
                finally:
                    # 关闭同步生成器会执行清理代码，并释放生成锁。
                    close = getattr(events, "close", None)
                    if close is not None:
                        await _run_sync_step(worker, close)
            finally:
                # 执行和清理都完成后，再关闭线程池。
                worker.shutdown(wait=True, cancel_futures=True)
