"""按模型迭代驱动 raw batching 与 continuous batching。

这个模块只实现异步 Engine 编排，不知道具体模型、PyTorch 或 HTTP：

- ``Scheduler`` 决定本轮有哪些请求进入执行批次；
- ``BatchTokenExecutor`` 为批次中的每个请求计算一个 token；
- Engine 更新请求状态，并把领域事件送回各自的异步流。

所有可变请求状态都由 ``_lock`` 保护，耗时的同步模型调用则在锁外执行。
这样新请求和取消操作不必等待当前模型迭代结束，同时执行结果仍能在锁内
按 request ID 安全地写回。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
from dataclasses import dataclass, field
from itertools import count

from light_vllm.execution.api import (
    BatchTokenExecutor,
    ExecutionBatch,
    ExecutionError,
    ExecutionNotReadyError,
    SequenceTokens,
    TokenSelection,
)
from light_vllm.generation.api import (
    FinishReason,
    GenerateRequest,
    GenerateResult,
    GenerationError,
    GenerationEvent,
    GenerationFinished,
    GenerationNotReadyError,
    TokenGenerated,
)
from light_vllm.scheduler.api import Scheduler, SchedulerError


@dataclass(frozen=True, slots=True)
class _RequestFailed:
    """放进请求事件队列的失败消息。

    ``asyncio.Queue`` 只能传值，不能直接抛异常，所以 driver 先把领域异常
    包装成队列消息，再由对应的 ``stream`` 在消费者协程中重新抛出。
    """

    error: GenerationError


_QueueItem = GenerationEvent | _RequestFailed


@dataclass(slots=True)
class _RequestState:
    """一个请求由 Engine 独占的可变生成状态。

    ``token_ids`` 始终包含原始 prompt 和已经生成的 token；
    ``generated_count`` 只统计新增 token，用于事件位置和长度停止条件。
    每个请求有独立事件队列，因此一次批量执行可以把结果分别送给多个消费者。
    队列当前有意不设容量，避免慢消费者阻塞整个执行批次；生产级背压应和
    admission、请求上限一起设计，不能在这里临时增加跨请求阻塞。
    """

    request_id: str
    request: GenerateRequest
    token_ids: list[int]
    generated_count: int = 0
    events: asyncio.Queue[_QueueItem] = field(default_factory=asyncio.Queue)


def _execution_error(exc: Exception) -> GenerationError:
    """把执行层异常收敛成 EngineClient 对外承诺的生成异常。"""

    if isinstance(exc, ExecutionNotReadyError):
        return GenerationNotReadyError("load a model before generating")
    if isinstance(exc, ExecutionError):
        return GenerationError(str(exc))
    return GenerationError(f"batch execution failed: {type(exc).__name__}: {exc}")


def _validated_selections(
    batch: ExecutionBatch,
    selections: tuple[TokenSelection, ...],
) -> dict[str, TokenSelection]:
    """校验批量执行结果与输入请求严格一一对应。

    Engine 按 request ID 写回结果，不依赖执行器保留输入顺序。重复、缺失或
    额外结果都会让整个执行批次失败，避免 token 被写到错误请求。
    """

    if any(not isinstance(selection, TokenSelection) for selection in selections):
        raise ExecutionError("batch executor must return TokenSelection values")

    by_request_id = {selection.request_id: selection for selection in selections}
    expected_ids = tuple(sequence.request_id for sequence in batch.sequences)
    if len(by_request_id) != len(selections) or set(by_request_id) != set(expected_ids):
        raise ExecutionError("batch executor must return one token for every request")
    return by_request_id


async def _await_safe_boundary(task: asyncio.Task[None]) -> None:
    """收到外层取消后仍等待内部任务到达安全边界。

    ``asyncio.shield`` 防止取消直接传播给清理任务或 driver；如果调用方已经
    被取消，仍先等内部任务结束，再把 ``CancelledError`` 交还给调用方。
    """

    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        with suppress(Exception):
            await task
        raise


class IterationBatchEngine:
    """在同一进程内按迭代执行调度批次。

    Engine 只维护请求状态、事件流和安全取消。每轮选择哪些请求由
    ``Scheduler`` 决定，如何执行批次由 ``BatchTokenExecutor`` 决定。

    同一实例只启动一个 driver，因此不会并发调用执行器。driver 空闲时自动
    退出；后续请求到达时再按需启动，不保留永久忙等的后台循环。
    """

    def __init__(self, executor: BatchTokenExecutor, scheduler: Scheduler) -> None:
        self._executor = executor
        self._scheduler = scheduler
        # Scheduler 中的每个 request ID 都必须在这里有且仅有一个状态对象。
        self._states: dict[str, _RequestState] = {}
        # ID 只用于当前 Engine 内部路由，不泄漏到生成或 HTTP 契约。
        self._request_ids = count(1)
        # 同时保护 _states、Scheduler 状态、_driver_task 和 _closed。
        # 模型执行不得持有这把锁。
        self._lock = asyncio.Lock()
        self._driver_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def ready(self) -> bool:
        return not self._closed and self._executor.ready

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        """收集同一个异步事件流，返回完整结果。"""

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
        """提交请求，并逐个返回该请求自己的生成事件。

        driver 可以快于消费者继续生成；事件通过请求自己的队列解耦。无论是
        正常结束、消费者主动关闭还是协程取消，``finally`` 都会执行幂等清理。
        """

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
            # 正常完成时状态已被 _finish_locked 移除；提前离开时这里负责取消。
            cleanup = asyncio.create_task(self._cancel(state.request_id))
            await _await_safe_boundary(cleanup)

    async def close(self) -> None:
        """停止接收请求，并等待正在执行的模型迭代到达安全边界。

        Python 不能强制停止 ``to_thread`` 中的同步模型调用，所以这里只移除
        所有调度状态并通知消费者失败，不取消 driver。当前模型步骤自然返回后，
        driver 会发现请求已不存在并安全退出。
        """

        async with self._lock:
            if self._closed:
                task = self._driver_task
            else:
                self._closed = True
                # 在同一个临界区内同时清空 Scheduler 和 Engine 状态，避免二者失配。
                for request_id, state in tuple(self._states.items()):
                    self._scheduler.remove(request_id)
                    state.events.put_nowait(
                        _RequestFailed(GenerationError("generation engine is closed"))
                    )
                self._states.clear()
                task = self._driver_task

        if task is not None:
            await _await_safe_boundary(task)

    async def _register(self, request: GenerateRequest) -> _RequestState:
        """原子地登记 Engine 状态和 Scheduler 状态。"""

        async with self._lock:
            if self._closed:
                raise GenerationError("generation engine is closed")
            if not self._executor.ready:
                raise GenerationNotReadyError("load a model before generating")

            request_id = f"request-{next(self._request_ids)}"
            state = _RequestState(
                request_id=request_id,
                request=request,
                token_ids=list(request.input_ids),
            )
            self._states[request_id] = state
            try:
                # Scheduler.add 失败时必须回滚刚写入的 Engine 状态。
                self._scheduler.add(request_id)
            except Exception:
                del self._states[request_id]
                raise
            self._start_driver_locked()
            return state

    async def _cancel(self, request_id: str) -> None:
        """幂等移除请求；正在执行的批次结果稍后会按 ID 自动丢弃。"""

        async with self._lock:
            state = self._states.pop(request_id, None)
            if state is not None:
                self._scheduler.remove(request_id)

    def _start_driver_locked(self) -> None:
        """按需启动唯一 driver；调用方必须已经持有 ``_lock``。"""

        if self._driver_task is None:
            self._driver_task = asyncio.create_task(
                self._drive(),
                name="light-vllm-iteration-engine",
            )

    async def _drive(self) -> None:
        """循环执行 schedule、execute、update 三个阶段。"""

        current_task = asyncio.current_task()
        try:
            while True:
                # 给同一轮事件循环中新到达的请求一次进入等待队列的机会。
                await asyncio.sleep(0)
                async with self._lock:
                    if not self._scheduler.has_requests:
                        self._driver_task = None
                        return

                    scheduled = self._scheduler.schedule()
                    if not scheduled.request_ids:
                        raise SchedulerError("scheduler returned an empty batch while work remains")
                    # 在锁内冻结本轮输入快照。释放锁后，新请求可以登记、旧请求
                    # 可以取消，但本轮同步执行看到的 token 序列保持不变。
                    execution_batch = ExecutionBatch(
                        sequences=tuple(
                            SequenceTokens(
                                request_id=request_id,
                                token_ids=tuple(self._states[request_id].token_ids),
                            )
                            for request_id in scheduled.request_ids
                        )
                    )

                try:
                    # 同步模型计算放在线程中，避免阻塞 HTTP 所在的 event loop。
                    # 唯一 driver 保证任意时刻最多只有一次 executor 调用。
                    raw_selections = await asyncio.to_thread(
                        self._executor.next_tokens,
                        execution_batch,
                    )
                    selections = _validated_selections(
                        execution_batch,
                        tuple(raw_selections),
                    )
                except Exception as exc:
                    # 单个执行批次失败不应杀死 Engine；等待队列仍可进入下一轮。
                    async with self._lock:
                        self._fail_batch_locked(scheduled.request_ids, exc)
                    continue

                async with self._lock:
                    # 写回时重新检查 request ID；执行期间取消的请求会被跳过。
                    self._apply_selections_locked(scheduled.request_ids, selections)
        except Exception as exc:
            # Scheduler/driver 自身的不变量失败会影响所有请求，统一广播错误。
            async with self._lock:
                self._fail_all_locked(exc)
        finally:
            async with self._lock:
                # 处理“driver 正准备退出时恰好有新请求到达”的竞态。若当前任务
                # 仍是登记中的 driver 且队列又有工作，就在锁内交接给新 driver。
                if self._driver_task is current_task:
                    self._driver_task = None
                    if self._scheduler.has_requests and not self._closed:
                        self._start_driver_locked()

    def _apply_selections_locked(
        self,
        request_ids: tuple[str, ...],
        selections: dict[str, TokenSelection],
    ) -> None:
        """把一轮 token 写回活动请求；调用方必须持有 ``_lock``。"""

        for request_id in request_ids:
            state = self._states.get(request_id)
            if state is None:
                # 请求可能在模型执行期间被取消。
                continue

            token_id = selections[request_id].token_id
            position = state.generated_count
            state.token_ids.append(token_id)
            state.generated_count += 1
            state.events.put_nowait(TokenGenerated(token_id=token_id, position=position))

            # token 事件必须先于 terminal 事件入队，消费者才能收集完整结果。
            if state.request.eos_token_id is not None and token_id == state.request.eos_token_id:
                self._finish_locked(state, "eos")
            elif state.generated_count == state.request.max_new_tokens:
                self._finish_locked(state, "length")

    def _finish_locked(self, state: _RequestState, finish_reason: FinishReason) -> None:
        """结束请求并发出唯一 terminal 事件；调用方必须持有 ``_lock``。"""

        # 先从调度状态中移除，确保下一轮能立即补位；事件队列由 stream
        # 持有，即使状态字典删除后仍可把 terminal 事件交给消费者。
        self._scheduler.remove(state.request_id)
        self._states.pop(state.request_id, None)
        state.events.put_nowait(GenerationFinished(finish_reason=finish_reason))

    def _fail_batch_locked(self, request_ids: tuple[str, ...], exc: Exception) -> None:
        """只终止本轮执行涉及的请求；调用方必须持有 ``_lock``。"""

        for request_id in request_ids:
            state = self._states.pop(request_id, None)
            self._scheduler.remove(request_id)
            if state is not None:
                state.events.put_nowait(_RequestFailed(_execution_error(exc)))

    def _fail_all_locked(self, exc: Exception) -> None:
        """在 driver 级故障时终止所有请求；调用方必须持有 ``_lock``。"""

        for request_id, state in tuple(self._states.items()):
            self._scheduler.remove(request_id)
            state.events.put_nowait(_RequestFailed(_execution_error(exc)))
        self._states.clear()
