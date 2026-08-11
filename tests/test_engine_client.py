from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import suppress
from threading import Event, Lock

from light_vllm import (
    GenerateRequest,
    GenerateResult,
    GenerationEvent,
    GenerationFinished,
    InProcessEngineClient,
    TokenGenerated,
)


class CloseTrackingIterator(Iterator[GenerationEvent]):
    def __init__(self) -> None:
        self.closed = False
        self._position = 0

    def __next__(self) -> GenerationEvent:
        if self._position == 0:
            event: GenerationEvent = TokenGenerated(token_id=7, position=0)
        elif self._position == 1:
            event = GenerationFinished(finish_reason="length")
        else:
            raise StopIteration
        self._position += 1
        return event

    def close(self) -> None:
        self.closed = True


class StubGenerationService:
    ready = True

    def __init__(self) -> None:
        self.events = CloseTrackingIterator()

    def stream(self, request: GenerateRequest) -> Iterator[GenerationEvent]:
        return self.events

    def generate(self, request: GenerateRequest) -> GenerateResult:
        return GenerateResult(
            input_ids=request.input_ids,
            generated_token_ids=(7,),
            finish_reason="length",
        )


def test_in_process_client_adapts_non_streaming_generation() -> None:
    service = StubGenerationService()
    client = InProcessEngineClient(service)

    result = asyncio.run(client.generate(GenerateRequest(input_ids=(1,), max_new_tokens=1)))

    assert client.ready
    assert result.generated_token_ids == (7,)


def test_in_process_client_closes_sync_stream_when_async_consumer_stops() -> None:
    service = StubGenerationService()
    client = InProcessEngineClient(service)

    async def consume_one_event() -> None:
        events = client.stream(GenerateRequest(input_ids=(1,), max_new_tokens=2))
        assert await anext(events) == TokenGenerated(token_id=7, position=0)
        await events.aclose()

    asyncio.run(consume_one_event())

    assert service.events.closed


def test_cancellation_waits_for_the_active_sync_step_before_closing() -> None:
    started = Event()
    release = Event()

    class BlockingIterator(CloseTrackingIterator):
        def __init__(self) -> None:
            super().__init__()
            self.executing = False
            self.closed_while_executing = False

        def __next__(self) -> GenerationEvent:
            self.executing = True
            started.set()
            release.wait()
            self.executing = False
            return super().__next__()

        def close(self) -> None:
            self.closed_while_executing = self.executing
            super().close()

    service = StubGenerationService()
    service.events = BlockingIterator()
    client = InProcessEngineClient(service)

    async def cancel_active_step() -> None:
        events = client.stream(GenerateRequest(input_ids=(1,), max_new_tokens=1))
        pending = asyncio.create_task(anext(events))

        try:
            assert await asyncio.wait_for(
                asyncio.to_thread(started.wait, 1.0),
                timeout=1.5,
            )
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
        finally:
            # 无论断言是否通过，都要释放工作线程，否则 asyncio.run() 关闭时可能卡住。
            release.set()
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(pending, timeout=1.0)

    asyncio.run(cancel_active_step())

    assert service.events.closed
    assert not service.events.closed_while_executing


def test_concurrent_streams_wait_without_occupying_worker_threads() -> None:
    entered = Event()
    release = Event()

    class ContendedIterator(Iterator[GenerationEvent]):
        def __init__(self, service: ContendedService) -> None:
            self._service = service
            self._emitted = False

        def __next__(self) -> GenerationEvent:
            if self._emitted:
                raise StopIteration
            self._emitted = True
            with self._service.state_lock:
                self._service.active_steps += 1
                self._service.max_active_steps = max(
                    self._service.max_active_steps,
                    self._service.active_steps,
                )
                entered.set()
            try:
                release.wait()
                return GenerationFinished(finish_reason="length")
            finally:
                with self._service.state_lock:
                    self._service.active_steps -= 1

    class ContendedService(StubGenerationService):
        def __init__(self) -> None:
            self.state_lock = Lock()
            self.active_steps = 0
            self.max_active_steps = 0

        def stream(self, request: GenerateRequest) -> Iterator[GenerationEvent]:
            return ContendedIterator(self)

    service = ContendedService()
    client = InProcessEngineClient(service)

    async def run_concurrent_streams() -> None:
        streams = [
            client.stream(GenerateRequest(input_ids=(token_id,), max_new_tokens=1))
            for token_id in range(3)
        ]
        pending = [asyncio.create_task(anext(events)) for events in streams]

        try:
            assert await asyncio.wait_for(
                asyncio.to_thread(entered.wait, 1.0),
                timeout=1.5,
            )
            await asyncio.sleep(0)
            assert service.max_active_steps == 1
        finally:
            release.set()
            for task in pending[1:]:
                task.cancel()
            for task in pending:
                with suppress(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1.0)
            for events in streams:
                await events.aclose()

    asyncio.run(run_concurrent_streams())

    assert service.max_active_steps == 1
