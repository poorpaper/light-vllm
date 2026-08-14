from __future__ import annotations

import asyncio
from contextlib import suppress
from threading import Event

import pytest

from light_vllm import GenerateRequest, GenerationError
from light_vllm.runtime.engine import EngineCore
from light_vllm.runtime.execution import (
    ExecutionBatch,
    ExecutionError,
    ExecutionOutput,
    RequestOutput,
)
from light_vllm.runtime.kv_cache import PagedKVCacheManager
from light_vllm.runtime.scheduler import TokenBudgetScheduler


class _Lease:
    def release(self) -> None:
        pass


class RecordingExecutor:
    ready = True

    def __init__(
        self,
        *,
        multiple_tokens: bool = False,
        block_first_step: bool = False,
    ) -> None:
        self.history: list[tuple[tuple[int, ...], ...]] = []
        self.active: set[str] = set()
        self.multiple_tokens = multiple_tokens
        self.block_first_step = block_first_step
        self.first_step_started = Event()
        self.release_first_step = Event()

    def add_request(self, request_id: str, *, capacity: int) -> None:
        self.active.add(request_id)

    def free_request(self, request_id: str) -> bool:
        existed = request_id in self.active
        self.active.discard(request_id)
        return existed

    def acquire(self, request_ids: tuple[str, ...]) -> _Lease:
        return _Lease()

    def execute(self, batch: ExecutionBatch) -> ExecutionOutput:
        step = len(self.history)
        self.history.append(tuple(request.input_token_ids for request in batch.requests))
        if step == 0:
            self.first_step_started.set()
            if self.block_first_step:
                self.release_first_step.wait()
        results = []
        for request in batch.requests:
            token_ids = ()
            if request.sampling_required:
                token_ids = (
                    (request.input_token_ids[-1] + 1, request.input_token_ids[-1] + 2)
                    if self.multiple_tokens
                    else (request.input_token_ids[-1] + 1,)
                )
            results.append(
                RequestOutput(
                    request_id=request.request_id,
                    num_computed_tokens=len(request.input_token_ids),
                    token_ids=token_ids,
                )
            )
        return ExecutionOutput(requests=tuple(results))


def _engine(executor: RecordingExecutor, *, token_budget: int = 2) -> EngineCore:
    scheduler = TokenBudgetScheduler(
        PagedKVCacheManager(num_blocks=16, block_size=2),
        max_num_sequences=2,
        max_num_scheduled_tokens=token_budget,
    )
    return EngineCore(executor, scheduler)


def test_engine_chunks_prefill_then_streams_generated_tokens() -> None:
    async def run() -> None:
        executor = RecordingExecutor()
        engine = _engine(executor, token_budget=2)
        result = await engine.generate(GenerateRequest(input_ids=(1, 2, 3), max_new_tokens=2))
        await engine.close()

        assert executor.history == [((1, 2),), ((3,),), ((4,),)]
        assert result.generated_token_ids == (4, 5)
        assert not executor.active

    asyncio.run(run())


def test_engine_accepts_multiple_committed_tokens_from_one_execution() -> None:
    async def run() -> None:
        engine = _engine(RecordingExecutor(multiple_tokens=True), token_budget=8)
        result = await engine.generate(GenerateRequest(input_ids=(1,), max_new_tokens=3))
        await engine.close()

        assert result.generated_token_ids == (2, 3, 4)
        assert result.finish_reason == "length"

    asyncio.run(run())


def test_cancelled_request_releases_resources_at_the_safe_boundary() -> None:
    async def run() -> None:
        executor = RecordingExecutor(block_first_step=True)
        kv_cache = PagedKVCacheManager(num_blocks=16, block_size=2)
        scheduler = TokenBudgetScheduler(
            kv_cache,
            max_num_sequences=2,
            max_num_scheduled_tokens=2,
        )
        engine = EngineCore(executor, scheduler)
        events = engine.stream(GenerateRequest(input_ids=(1,), max_new_tokens=2))
        pending = asyncio.create_task(anext(events))
        try:
            assert await asyncio.wait_for(
                asyncio.to_thread(executor.first_step_started.wait, 1.0),
                timeout=1.5,
            )
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending
            assert not executor.active
            assert kv_cache.num_free_blocks == 15
        finally:
            executor.release_first_step.set()
        await events.aclose()
        await engine.close()
        assert kv_cache.num_free_blocks == 16

    asyncio.run(run())


def test_execution_failure_is_delivered_to_the_request() -> None:
    class FailingExecutor(RecordingExecutor):
        def execute(self, batch: ExecutionBatch) -> ExecutionOutput:
            raise ExecutionError("invalid execution output")

    async def run() -> None:
        engine = _engine(FailingExecutor())
        with pytest.raises(GenerationError, match="invalid execution output"):
            await engine.generate(GenerateRequest(input_ids=(1,), max_new_tokens=1))
        await engine.close()

    asyncio.run(run())
