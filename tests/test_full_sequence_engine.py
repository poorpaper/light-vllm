from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from threading import Event, Lock

import pytest

from light_vllm import FullSequenceBatchEngine, GenerateRequest, GenerationError
from light_vllm.execution import ExecutionBatch, ExecutionError, TokenSelection
from light_vllm.scheduler import ContinuousBatchScheduler, Scheduler, StaticBatchScheduler


class RecordingBatchExecutor:
    ready = True

    def __init__(self, *, block_first_step: bool = False) -> None:
        self.history: list[tuple[tuple[int, ...], ...]] = []
        self.first_step_started = Event()
        self.release_first_step = Event()
        self._block_first_step = block_first_step
        self._lock = Lock()

    def next_tokens(self, batch: ExecutionBatch) -> tuple[TokenSelection, ...]:
        with self._lock:
            step = len(self.history)
            self.history.append(tuple(sequence.token_ids for sequence in batch.sequences))
        if step == 0:
            self.first_step_started.set()
            if self._block_first_step:
                self.release_first_step.wait()
        return tuple(
            TokenSelection(
                request_id=sequence.request_id,
                token_id=sequence.token_ids[-1] + 1,
            )
            for sequence in batch.sequences
        )


def _run_with_late_request(
    scheduler_factory: Callable[[], Scheduler],
) -> tuple[tuple[tuple[tuple[int, ...], ...], ...], tuple[tuple[int, ...], tuple[int, ...]]]:
    async def run() -> tuple[
        tuple[tuple[tuple[int, ...], ...], ...],
        tuple[tuple[int, ...], tuple[int, ...]],
    ]:
        executor = RecordingBatchExecutor(block_first_step=True)
        engine = FullSequenceBatchEngine(executor, scheduler_factory())
        first = asyncio.create_task(
            engine.generate(GenerateRequest(input_ids=(1,), max_new_tokens=3))
        )

        assert await asyncio.wait_for(
            asyncio.to_thread(executor.first_step_started.wait, 1.0),
            timeout=1.5,
        )
        second = asyncio.create_task(
            engine.generate(GenerateRequest(input_ids=(10,), max_new_tokens=1))
        )
        await asyncio.sleep(0)
        executor.release_first_step.set()

        try:
            first_result, second_result = await asyncio.wait_for(
                asyncio.gather(first, second),
                timeout=2.0,
            )
        finally:
            executor.release_first_step.set()
            await engine.close()

        return (
            tuple(executor.history),
            (first_result.generated_token_ids, second_result.generated_token_ids),
        )

    return asyncio.run(run())


def test_continuous_and_static_batching_keep_results_equal_but_refill_differently() -> None:
    continuous_history, continuous_results = _run_with_late_request(
        lambda: ContinuousBatchScheduler(max_num_sequences=2)
    )
    static_history, static_results = _run_with_late_request(
        lambda: StaticBatchScheduler(max_num_sequences=2)
    )

    assert continuous_results == static_results == ((2, 3, 4), (11,))
    assert continuous_history == (
        ((1,),),
        ((1, 2), (10,)),
        ((1, 2, 3),),
    )
    assert static_history == (
        ((1,),),
        ((1, 2),),
        ((1, 2, 3),),
        ((10,),),
    )


def test_cancelled_request_is_removed_at_the_next_safe_iteration_boundary() -> None:
    async def run() -> None:
        executor = RecordingBatchExecutor(block_first_step=True)
        engine = FullSequenceBatchEngine(
            executor,
            ContinuousBatchScheduler(max_num_sequences=1),
        )
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
        finally:
            executor.release_first_step.set()

        result = await asyncio.wait_for(
            engine.generate(GenerateRequest(input_ids=(10,), max_new_tokens=1)),
            timeout=2.0,
        )
        await events.aclose()
        await engine.close()

        assert result.generated_token_ids == (11,)
        assert all(sequence[0] != 1 for batch in executor.history[1:] for sequence in batch)

    asyncio.run(run())


def test_full_sequence_engine_applies_eos_stop_conditions() -> None:
    async def run() -> None:
        engine = FullSequenceBatchEngine(
            RecordingBatchExecutor(),
            ContinuousBatchScheduler(max_num_sequences=2),
        )
        result = await engine.generate(
            GenerateRequest(input_ids=(1,), max_new_tokens=3, eos_token_id=2)
        )
        await engine.close()

        assert result.generated_token_ids == (2,)
        assert result.finish_reason == "eos"

    asyncio.run(run())


def test_batch_execution_failure_is_delivered_to_the_request() -> None:
    class FailingBatchExecutor:
        ready = True

        def next_tokens(self, batch: ExecutionBatch) -> tuple[TokenSelection, ...]:
            raise ExecutionError("invalid batch output")

    async def run() -> None:
        engine = FullSequenceBatchEngine(
            FailingBatchExecutor(),
            ContinuousBatchScheduler(max_num_sequences=1),
        )
        with pytest.raises(GenerationError, match="invalid batch output"):
            await engine.generate(GenerateRequest(input_ids=(1,), max_new_tokens=1))
        await engine.close()

    asyncio.run(run())
