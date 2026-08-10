"""Serving-side engine boundary and the smallest in-process implementation.

The reference generator is intentionally synchronous so it stays easy to run,
profile, and compare with future implementations. Serving adapters use the
asynchronous ``EngineClient`` boundary instead, so a later scheduled or
out-of-process engine does not require HTTP/RPC changes.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import suppress
from typing import Protocol, TypeVar

from light_vllm.contracts import (
    GenerateRequest,
    GenerateResult,
    GenerationError,
    GenerationEvent,
    GenerationFinished,
    GenerationService,
    TokenGenerated,
)


class EngineClient(Protocol):
    """Async port used by serving protocols to reach any inference engine.

    Implementations own cancellation and transport details. A protocol adapter
    therefore consumes events without knowing whether inference runs in a
    worker thread, an event loop, or another process.
    """

    @property
    def ready(self) -> bool: ...

    def stream(self, request: GenerateRequest) -> AsyncIterator[GenerationEvent]: ...

    async def generate(self, request: GenerateRequest) -> GenerateResult: ...


class _StreamEnded(Exception):
    """Thread-safe marker for a normally exhausted synchronous iterator."""

    pass


_ResultT = TypeVar("_ResultT")


def _next_event(events: Iterator[GenerationEvent]) -> GenerationEvent:
    # StopIteration cannot cross an asyncio Future boundary safely. Translate
    # normal iterator exhaustion inside the worker thread and handle it after
    # control returns to the event loop.
    try:
        return next(events)
    except StopIteration as exc:
        raise _StreamEnded from exc


async def _run_sync_step(
    worker: Executor,
    callback: Callable[[], _ResultT],
) -> _ResultT:
    """Run one sync step and do not leave it behind during cancellation.

    Python cannot cancel a function that is already running in a thread.
    Waiting for that function to reach a boundary before re-raising cancellation
    prevents iterator cleanup from racing with an active ``next()`` call.
    """

    loop = asyncio.get_running_loop()
    pending = loop.run_in_executor(worker, callback)
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        with suppress(Exception):
            await pending
        raise


class InProcessEngineClient:
    """Adapt a synchronous reference service to the async serving boundary.

    This class is a bridge, not a scheduler: it preserves the service's serial
    execution semantics and pulls exactly one domain event per async iteration.
    """

    def __init__(self, service: GenerationService) -> None:
        self._service = service
        # Queue concurrent serving requests in the event loop. Without this
        # gate, each waiter would occupy a worker thread while blocking on the
        # synchronous service lock; enough waiters could starve the active
        # request of the thread it needs for its next step or cleanup.
        self._admission_lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self._service.ready

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        """Collect the async event stream used by streaming transports."""

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
        """Expose a synchronous domain iterator as a cancellation-safe async stream."""

        async with self._admission_lock:
            # One single-thread executor belongs to one admitted stream. Every
            # stream call, ``next``, and ``close`` therefore runs on the same worker, which is
            # friendlier to thread-affine iterators and avoids relying on the
            # event loop's shared default executor for token-by-token progress.
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
                    # Closing the synchronous generator unwinds its ``with``/``finally``
                    # blocks, including the reference generator's execution lock.
                    close = getattr(events, "close", None)
                    if close is not None:
                        await _run_sync_step(worker, close)
            finally:
                # The active step and cleanup have both completed here, so
                # shutdown is immediate and cannot strand background work.
                worker.shutdown(wait=True, cancel_futures=True)
