from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from threading import Event

import pytest

from light_vllm import GenerateRequest, GenerationError
from light_vllm.runtime.engine import EngineCore
from light_vllm.runtime.execution import (
    ExecutionBatch,
    ExecutionCapabilities,
    ExecutionError,
    ExecutionOutput,
    RequestOutput,
)
from light_vllm.runtime.kv_cache import FixedKVBlockCapacity, PagedKVCacheManager
from light_vllm.runtime.scheduler import DecodingBudget, TokenBudgetScheduler


class _Lease:
    def __init__(self, on_release: Callable[[], None]) -> None:
        self._on_release = on_release
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._on_release()
            self._released = True


class RecordingExecutor:
    ready = True
    capabilities = ExecutionCapabilities(
        max_model_tokens=None,
        max_kv_cache_tokens=32,
    )

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
        self.lease_release_count = 0

    def add_request(self, request_id: str, *, capacity: int) -> None:
        self.active.add(request_id)

    def free_request(self, request_id: str) -> bool:
        existed = request_id in self.active
        self.active.discard(request_id)
        return existed

    def acquire(self, request_ids: tuple[str, ...]) -> _Lease:
        return _Lease(self._record_lease_release)

    def _record_lease_release(self) -> None:
        self.lease_release_count += 1

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
            if request.max_output_tokens:
                candidates = (
                    (request.input_token_ids[-1] + 1, request.input_token_ids[-1] + 2)
                    if self.multiple_tokens
                    else (request.input_token_ids[-1] + 1,)
                )
                token_ids = candidates[: request.max_output_tokens]
            results.append(
                RequestOutput(
                    request_id=request.request_id,
                    num_input_tokens_computed=len(request.input_token_ids),
                    output_token_ids=token_ids,
                    num_cached_output_tokens=(
                        1 if self.multiple_tokens and request.num_lookahead_tokens else 0
                    ),
                )
            )
        return ExecutionOutput(requests=tuple(results))


def _engine(executor: RecordingExecutor, *, token_budget: int = 2) -> EngineCore:
    scheduler = TokenBudgetScheduler(
        PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=16, block_size=2)),
        max_num_sequences=2,
        max_num_scheduled_tokens=token_budget,
        decoding_budget=(
            DecodingBudget(num_lookahead_tokens=1, max_output_tokens=2)
            if executor.multiple_tokens
            else None
        ),
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
        executor = RecordingExecutor(multiple_tokens=True)
        engine = _engine(executor, token_budget=8)
        result = await engine.generate(GenerateRequest(input_ids=(1,), max_new_tokens=3))
        await engine.close()

        assert result.generated_token_ids == (2, 3, 4)
        assert result.finish_reason == "length"
        assert executor.history == [((1,),), ((3,),)]

    asyncio.run(run())


def test_engine_stops_at_eos_inside_a_multi_token_result() -> None:
    async def run() -> None:
        executor = RecordingExecutor(multiple_tokens=True)
        engine = _engine(executor, token_budget=8)
        result = await engine.generate(
            GenerateRequest(input_ids=(1,), max_new_tokens=3, eos_token_id=2)
        )
        await engine.close()

        assert result.generated_token_ids == (2,)
        assert result.finish_reason == "eos"
        assert executor.history == [((1,),)]
        assert not executor.active

    asyncio.run(run())


def test_cancelled_request_releases_resources_at_the_safe_boundary() -> None:
    async def run() -> None:
        executor = RecordingExecutor(block_first_step=True)
        kv_cache = PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=16, block_size=2))
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
        assert executor.lease_release_count == 1

    asyncio.run(run())


def test_execution_failure_is_delivered_to_the_request() -> None:
    class FailingExecutor(RecordingExecutor):
        def execute(self, batch: ExecutionBatch) -> ExecutionOutput:
            raise ExecutionError("invalid execution output")

    async def run() -> None:
        executor = FailingExecutor()
        engine = _engine(executor)
        with pytest.raises(GenerationError, match="invalid execution output"):
            await engine.generate(GenerateRequest(input_ids=(1,), max_new_tokens=1))
        await engine.close()
        assert executor.lease_release_count == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("computed", "computed-token count"),
        ("too_many_outputs", "more tokens than the execution budget"),
        ("cached_beyond_lookahead", "cached more output tokens"),
        ("wrong_boundary", "wrong scheduling boundary"),
    ],
)
def test_engine_rejects_invalid_multi_token_execution_facts(
    case: str,
    expected_error: str,
) -> None:
    class InvalidExecutor(RecordingExecutor):
        def execute(self, batch: ExecutionBatch) -> ExecutionOutput:
            request = batch.requests[0]
            computed = len(request.input_token_ids)
            outputs: tuple[int, ...] = (2, 3)
            cached = 1
            if case == "computed":
                computed += 1
            elif case == "too_many_outputs":
                outputs = (2, 3, 4)
            elif case == "cached_beyond_lookahead":
                cached = 2
            elif case == "wrong_boundary":
                outputs = ()
                cached = 0
            return ExecutionOutput(
                requests=(
                    RequestOutput(
                        request_id=request.request_id,
                        num_input_tokens_computed=computed,
                        output_token_ids=outputs,
                        num_cached_output_tokens=cached,
                    ),
                )
            )

    async def run() -> None:
        executor = InvalidExecutor(multiple_tokens=True)
        kv_cache = PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=16, block_size=2))
        scheduler = TokenBudgetScheduler(
            kv_cache,
            max_num_sequences=2,
            max_num_scheduled_tokens=8,
            decoding_budget=DecodingBudget(num_lookahead_tokens=1, max_output_tokens=2),
        )
        engine = EngineCore(executor, scheduler)
        with pytest.raises(GenerationError, match=expected_error):
            await engine.generate(GenerateRequest(input_ids=(1,), max_new_tokens=3))
        await engine.close()

        assert not executor.active
        assert kv_cache.num_free_blocks == 16
        assert executor.lease_release_count == 1

    asyncio.run(run())
