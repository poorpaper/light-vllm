"""把同步模型执行固定到 Engine 私有线程。"""

from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import cast

from light_vllm.runtime.execution.interfaces import (
    ExecutionBatch,
    ExecutionOutput,
    ModelExecutor,
)


@dataclass(frozen=True, slots=True)
class _LaneWork:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[ExecutionOutput]
    batch: ExecutionBatch


_STOP = object()


class ExecutionLane:
    """在一个常驻线程中串行执行单个 Engine 的模型步骤。

    Engine 当前只允许一个模型步骤在途。专用 lane 保留这个边界，同时避免
    每轮向进程级线程池创建并提交临时 work item。完成结果仍回到原事件循环，
    请求状态和 KV lease 继续只由 Engine Core 提交与释放。
    """

    def __init__(self, executor: ModelExecutor) -> None:
        self._executor = executor
        self._work: queue.SimpleQueue[_LaneWork | object] = queue.SimpleQueue()
        self._state_lock = threading.Lock()
        self._pending = False
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="light-vllm-execution-lane",
            daemon=True,
        )
        self._thread.start()

    async def execute(self, batch: ExecutionBatch) -> ExecutionOutput:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        with self._state_lock:
            if self._closed:
                raise RuntimeError("execution lane is closed")
            if self._pending:
                raise RuntimeError("execution lane already has a step in flight")
            self._pending = True
        self._work.put(_LaneWork(loop=loop, future=future, batch=batch))
        return await future

    def close(self) -> None:
        """等待在途步骤越过执行边界，然后回收私有线程。"""

        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._work.put(_STOP)
        self._thread.join()

    def _run(self) -> None:
        while True:
            work = self._work.get()
            if work is _STOP:
                return
            assert isinstance(work, _LaneWork)
            try:
                output = self._executor.execute(work.batch)
            except BaseException as exc:
                self._notify(work, error=exc)
            else:
                self._notify(work, output=output)

    def _notify(
        self,
        work: _LaneWork,
        *,
        output: ExecutionOutput | None = None,
        error: BaseException | None = None,
    ) -> None:
        try:
            work.loop.call_soon_threadsafe(
                self._finish,
                work.future,
                output,
                error,
            )
        except RuntimeError:
            # 事件循环异常关闭时也要解除 lane 的占用，close() 才能安全结束。
            with self._state_lock:
                self._pending = False

    def _finish(
        self,
        future: asyncio.Future[ExecutionOutput],
        output: ExecutionOutput | None,
        error: BaseException | None,
    ) -> None:
        with self._state_lock:
            self._pending = False
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
            return
        # Executor 的返回类型仍由 Engine 的边界校验负责；这里原样传递错误实现的 None，
        # 让请求得到明确的 ExecutionError，而不是让等待方悬挂。
        future.set_result(cast(ExecutionOutput, output))
