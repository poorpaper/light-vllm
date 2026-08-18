"""CPU 与 CUDA 执行步骤的完成时间测量。"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import torch

from light_vllm.runtime.execution.interfaces import ExecutionOutput


class WallClockExecutionTimer:
    """同步 CPU 步骤使用单调墙钟计时。"""

    def __init__(self, clock: Callable[[], float] = perf_counter) -> None:
        self._clock = clock

    def measure(
        self,
        operation: Callable[[], ExecutionOutput],
    ) -> tuple[ExecutionOutput, float]:
        started_at = self._clock()
        output = operation()
        return output, self._clock() - started_at


class CudaEventExecutionTimer:
    """用 CUDA event 测量当前 stream 上本轮实际完成时间。"""

    def __init__(self, device: str | torch.device) -> None:
        self._device = torch.device(device)
        if self._device.type != "cuda":
            raise ValueError("CUDA event timing requires a CUDA device")

    def measure(
        self,
        operation: Callable[[], ExecutionOutput],
    ) -> tuple[ExecutionOutput, float]:
        with torch.cuda.device(self._device):
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            output = operation()
            finished.record()
            # 指标需要设备真实完成时间；等待只留在 CUDA Executor 边界。
            finished.synchronize()
            elapsed_seconds = started.elapsed_time(finished) / 1000.0
        return output, elapsed_seconds
