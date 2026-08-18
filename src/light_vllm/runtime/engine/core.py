"""驱动 token-budget 调度与模型执行的进程内 Engine Core。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
from dataclasses import dataclass, field
from itertools import count

from light_vllm.runtime.engine.admission import CapacityAdmission, SafeTTFTAdmission
from light_vllm.runtime.engine.interfaces import (
    EngineCapabilities,
    RequestAdmission,
    TTFTAdmission,
)
from light_vllm.runtime.execution.interfaces import (
    ExecutionBatch,
    ExecutionError,
    ExecutionNotReadyError,
    ExecutionOutput,
    ExecutionRequest,
    ModelExecutor,
    RequestOutput,
)
from light_vllm.runtime.generation.interfaces import (
    FinishReason,
    GenerateRequest,
    GenerateResult,
    GenerationError,
    GenerationEvent,
    GenerationFinished,
    GenerationNotReadyError,
    GenerationOverloadedError,
    GenerationRejectedError,
    TokenGenerated,
)
from light_vllm.runtime.observability.dispatch import SafeCompositePerformanceObserver
from light_vllm.runtime.observability.interfaces import (
    PerformanceObserver,
    StepObservation,
)
from light_vllm.runtime.scheduler.interfaces import Scheduler, SchedulerError, SchedulerOutput


@dataclass(frozen=True, slots=True)
class _RequestFailed:
    error: GenerationError


_QueueItem = GenerationEvent | _RequestFailed


@dataclass(slots=True)
class _RequestState:
    request_id: str
    request: GenerateRequest
    token_ids: list[int]
    generated_count: int = 0
    events: asyncio.Queue[_QueueItem] = field(default_factory=asyncio.Queue)


def _execution_error(exc: Exception) -> GenerationError:
    if isinstance(exc, ExecutionNotReadyError):
        return GenerationNotReadyError("load a model before generating")
    if isinstance(exc, ExecutionError):
        return GenerationError(str(exc))
    return GenerationError(f"model execution failed: {type(exc).__name__}: {exc}")


def _validated_output(batch: ExecutionBatch, output: ExecutionOutput) -> dict[str, RequestOutput]:
    """先检查执行结果是否与本轮输入和预算一致，再更新请求状态。"""

    if not isinstance(output, ExecutionOutput):
        raise ExecutionError("model executor must return ExecutionOutput")
    by_request_id = {result.request_id: result for result in output.requests}
    if set(by_request_id) != set(batch.request_ids):
        raise ExecutionError("model executor must return one result for every request")
    for request in batch.requests:
        result = by_request_id[request.request_id]
        if result.num_input_tokens_computed != len(request.input_token_ids):
            raise ExecutionError("executor computed-token count does not match its input")
        if len(result.output_token_ids) > request.max_output_tokens:
            raise ExecutionError("executor returned more tokens than the execution budget")
        if bool(result.output_token_ids) != bool(request.max_output_tokens):
            raise ExecutionError("executor returned tokens at the wrong scheduling boundary")
        if result.num_cached_output_tokens > request.num_lookahead_tokens:
            raise ExecutionError("executor cached more output tokens than reserved lookahead")
    return by_request_id


async def _await_safe_boundary(task: asyncio.Task[None]) -> None:
    """即使调用方取消，也要等清理任务结束，避免缓存仍在使用时被释放。"""

    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        with suppress(Exception):
            await task
        raise


class EngineCore:
    """按 ``schedule → execute → update`` 驱动所有活动请求。

    Engine 保存请求和输出事件；Scheduler 决定哪些请求本轮运行并分配 KV；
    Executor 负责张量和模型计算。三者只交换本轮输入和结果，不直接修改
    对方内部状态。
    """

    def __init__(
        self,
        executor: ModelExecutor,
        scheduler: Scheduler,
        admission: RequestAdmission | None = None,
        performance_observer: PerformanceObserver | None = None,
        ttft_admission: TTFTAdmission | None = None,
    ) -> None:
        self._executor = executor
        self._scheduler = scheduler
        self._admission = admission or CapacityAdmission()
        self._ttft_admission = SafeTTFTAdmission(ttft_admission)
        self._performance_observer = SafeCompositePerformanceObserver(performance_observer)
        self._states: dict[str, _RequestState] = {}
        self._executing_request_ids: set[str] = set()
        self._pending_scheduler_removals: set[str] = set()
        self._request_ids = count(1)
        self._lock = asyncio.Lock()
        self._driver_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def ready(self) -> bool:
        return not self._closed and self._executor.ready

    @property
    def capabilities(self) -> EngineCapabilities:
        execution = self._executor.capabilities
        return EngineCapabilities(
            max_model_tokens=execution.max_model_tokens,
            max_kv_cache_tokens=execution.max_kv_cache_tokens,
            max_num_sequences=self._scheduler.max_num_sequences,
            max_num_scheduled_tokens=self._scheduler.max_num_scheduled_tokens,
        )

    async def generate(self, request: GenerateRequest) -> GenerateResult:
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
        state = await self._register(request)
        try:
            while True:
                item = await state.events.get()
                if isinstance(item, _RequestFailed):
                    raise item.error
                yield item
                if isinstance(item, GenerationFinished):
                    return
        finally:
            cleanup = asyncio.create_task(self._cancel(state.request_id))
            await _await_safe_boundary(cleanup)

    async def close(self) -> None:
        async with self._lock:
            if not self._closed:
                self._closed = True
                for request_id, state in tuple(self._states.items()):
                    removed = self._remove_request_locked(request_id)
                    if removed is not None:
                        self._performance_observer.request_finished(
                            request_id,
                            outcome="cancelled",
                        )
                    state.events.put_nowait(
                        _RequestFailed(GenerationError("generation engine is closed"))
                    )
                self._publish_scheduler_stats()
            task = self._driver_task
        if task is not None:
            await _await_safe_boundary(task)

    async def _register(self, request: GenerateRequest) -> _RequestState:
        async with self._lock:
            if self._closed:
                raise GenerationError("generation engine is closed")
            if not self._executor.ready:
                raise GenerationNotReadyError("load a model before generating")
            # 请求即使独占引擎也装不下时立即拒绝；暂时没资源则进入调度等待。
            try:
                self._admission.validate(request, self.capabilities)
            except GenerationRejectedError:
                self._performance_observer.request_rejected(reason="capacity")
                raise
            try:
                self._ttft_admission.validate(request, self._scheduler.stats)
            except GenerationOverloadedError:
                self._performance_observer.request_rejected(reason="overloaded")
                raise

            request_id = f"request-{next(self._request_ids)}"
            state = _RequestState(request_id, request, list(request.input_ids))
            self._states[request_id] = state
            capacity = len(request.input_ids) + request.max_new_tokens
            try:
                self._executor.add_request(request_id, capacity=capacity)
                self._scheduler.add(
                    request_id,
                    token_ids=request.input_ids,
                    max_num_tokens=capacity,
                    cache_epoch=self._executor.capabilities.kv_cache_epoch,
                )
            except Exception:
                self._executor.free_request(request_id)
                self._scheduler.remove(request_id)
                self._states.pop(request_id, None)
                raise
            self._performance_observer.request_started(
                request_id,
                num_prompt_tokens=len(request.input_ids),
            )
            self._publish_scheduler_stats()
            self._start_driver_locked()
            return state

    async def _cancel(self, request_id: str) -> None:
        async with self._lock:
            state = self._remove_request_locked(request_id)
            if state is not None:
                self._performance_observer.request_finished(request_id, outcome="cancelled")
                self._publish_scheduler_stats()

    def _start_driver_locked(self) -> None:
        if self._driver_task is None:
            self._driver_task = asyncio.create_task(
                self._drive(),
                name="light-vllm-engine-core",
            )

    async def _drive(self) -> None:
        current_task = asyncio.current_task()
        try:
            while True:
                await asyncio.sleep(0)
                async with self._lock:
                    if not self._scheduler.has_requests:
                        self._driver_task = None
                        return
                    scheduled = self._scheduler.schedule()
                    self._publish_scheduler_stats()
                    batch = self._build_execution_batch_locked(scheduled)
                    lease = self._executor.acquire(batch.request_ids)
                    self._executing_request_ids.update(batch.request_ids)

                # 锁内只生成计划并保留缓存；耗时的模型计算在线程和锁外执行。
                try:
                    raw_output = await asyncio.to_thread(self._executor.execute, batch)
                    # 结果完整通过检查前，不更新请求进度，也不提交 KV cache。
                    output = _validated_output(batch, raw_output)
                    if raw_output.step_elapsed_seconds is not None:
                        observation = StepObservation(
                            num_scheduled_tokens=sum(
                                len(request.input_token_ids) + request.num_lookahead_tokens
                                for request in batch.requests
                            ),
                            num_requests=len(batch.requests),
                            elapsed_seconds=raw_output.step_elapsed_seconds,
                        )
                        # 控制组件和只读 Observer 消费同一个已完成 step 事实。
                        self._ttft_admission.step_completed(observation)
                        self._performance_observer.step_completed(observation)
                except Exception as exc:
                    async with self._lock:
                        self._fail_batch_locked(scheduled.request_ids, exc)
                    continue
                finally:
                    try:
                        # 等模型不再使用缓存后，调度器才可以回收对应 block。
                        lease.release()
                    finally:
                        async with self._lock:
                            self._finish_execution_locked(batch.request_ids)

                async with self._lock:
                    self._apply_output_locked(scheduled, output)
        except Exception as exc:
            async with self._lock:
                self._fail_all_locked(exc)
        finally:
            async with self._lock:
                if self._driver_task is current_task:
                    self._driver_task = None
                    if self._scheduler.has_requests and not self._closed:
                        self._start_driver_locked()

    def _build_execution_batch_locked(self, scheduled: SchedulerOutput) -> ExecutionBatch:
        requests: list[ExecutionRequest] = []
        for item in scheduled.requests:
            state = self._states[item.request_id]
            end = item.num_computed_tokens + item.num_scheduled_tokens
            # Scheduler 只给位置和数量，真实 token ID 由 Engine 请求状态切出。
            input_token_ids = tuple(state.token_ids[item.num_computed_tokens : end])
            if len(input_token_ids) != item.num_scheduled_tokens:
                raise SchedulerError("scheduler selected tokens outside the request state")
            requests.append(
                ExecutionRequest(
                    request_id=item.request_id,
                    input_token_ids=input_token_ids,
                    context_token_ids=tuple(state.token_ids),
                    num_computed_tokens=item.num_computed_tokens,
                    num_lookahead_tokens=item.num_lookahead_tokens,
                    max_output_tokens=item.max_output_tokens,
                    block_ids=item.block_ids,
                    num_readonly_prefix_blocks=item.num_readonly_prefix_blocks,
                )
            )
        return ExecutionBatch(requests=tuple(requests))

    def _apply_output_locked(
        self,
        scheduled: SchedulerOutput,
        output: dict[str, RequestOutput],
    ) -> None:
        for item in scheduled.requests:
            state = self._states.get(item.request_id)
            if state is None:
                continue
            result = output[item.request_id]
            # 一次返回多个 token 时，遇到 EOS 或长度上限就截断后面的结果。
            visible_tokens, finish_reason = self._visible_tokens(state, result.output_token_ids)
            num_cached_visible_tokens = min(
                result.num_cached_output_tokens,
                len(visible_tokens),
            )
            self._scheduler.complete(
                item.request_id,
                num_committed_tokens=(result.num_input_tokens_computed + num_cached_visible_tokens),
                num_new_tokens=len(visible_tokens),
            )
            if visible_tokens:
                self._performance_observer.tokens_generated(
                    item.request_id,
                    count=len(visible_tokens),
                )
            # 已确认但尚未写入 KV cache 的 token，会在下一轮作为输入再计算一次。
            for token_id in visible_tokens:
                position = state.generated_count
                state.token_ids.append(token_id)
                state.generated_count += 1
                state.events.put_nowait(TokenGenerated(token_id=token_id, position=position))
            if finish_reason is not None:
                self._finish_locked(state, finish_reason)
        self._publish_scheduler_stats()

    def _visible_tokens(
        self,
        state: _RequestState,
        token_ids: tuple[int, ...],
    ) -> tuple[tuple[int, ...], FinishReason | None]:
        visible: list[int] = []
        for token_id in token_ids:
            visible.append(token_id)
            if state.request.eos_token_id == token_id:
                return tuple(visible), "eos"
            if state.generated_count + len(visible) == state.request.max_new_tokens:
                return tuple(visible), "length"
        return tuple(visible), None

    def _finish_locked(self, state: _RequestState, reason: FinishReason) -> None:
        self._remove_request_locked(state.request_id)
        self._performance_observer.request_finished(state.request_id, outcome="finished")
        state.events.put_nowait(GenerationFinished(finish_reason=reason))

    def _remove_request_locked(self, request_id: str) -> _RequestState | None:
        state = self._states.pop(request_id, None)
        if state is not None:
            # 取消立即让请求离开 Engine 和 Worker，但执行中的 block table 必须
            # 保留到当前同步模型步骤结束，避免物理页被过早复用。
            if request_id in self._executing_request_ids:
                self._pending_scheduler_removals.add(request_id)
            else:
                self._scheduler.remove(request_id)
            self._executor.free_request(request_id)
        return state

    def _finish_execution_locked(self, request_ids: tuple[str, ...]) -> None:
        for request_id in request_ids:
            self._executing_request_ids.discard(request_id)
            if request_id in self._pending_scheduler_removals:
                self._pending_scheduler_removals.remove(request_id)
                self._scheduler.remove(request_id)
        self._publish_scheduler_stats()

    def _fail_batch_locked(self, request_ids: tuple[str, ...], exc: Exception) -> None:
        for request_id in request_ids:
            state = self._remove_request_locked(request_id)
            if state is not None:
                self._performance_observer.request_finished(request_id, outcome="failed")
                state.events.put_nowait(_RequestFailed(_execution_error(exc)))

    def _fail_all_locked(self, exc: Exception) -> None:
        for request_id in tuple(self._states):
            state = self._remove_request_locked(request_id)
            if state is not None:
                self._performance_observer.request_finished(request_id, outcome="failed")
                state.events.put_nowait(_RequestFailed(_execution_error(exc)))
        self._publish_scheduler_stats()

    def refresh_performance_metrics(self) -> None:
        """模型与 KV 规划完成后发布初始容量；不改变任何运行状态。"""

        self._publish_scheduler_stats()

    def _publish_scheduler_stats(self) -> None:
        self._performance_observer.scheduler_updated(self._scheduler.stats)
