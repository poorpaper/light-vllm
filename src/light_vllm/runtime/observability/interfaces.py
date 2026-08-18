from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Protocol, TypeAlias

from light_vllm.runtime.scheduler.interfaces import SchedulerStats

RequestOutcome: TypeAlias = Literal["finished", "failed", "cancelled"]
AdmissionRejection: TypeAlias = Literal["capacity", "overloaded"]


@dataclass(frozen=True, slots=True)
class HistogramSnapshot:
    """一个 Prometheus histogram 所需的累计事实。"""

    bounds: tuple[float, ...]
    cumulative_counts: tuple[int, ...]
    count: int
    total: float

    def __post_init__(self) -> None:
        if len(self.bounds) != len(self.cumulative_counts):
            raise ValueError("histogram bounds and counts must have the same length")
        if any(bound <= 0 for bound in self.bounds):
            raise ValueError("histogram bounds must be positive")
        if any(left >= right for left, right in zip(self.bounds, self.bounds[1:], strict=False)):
            raise ValueError("histogram bounds must be strictly increasing")
        if any(count < 0 for count in self.cumulative_counts):
            raise ValueError("histogram counts must be non-negative")
        if any(
            left > right
            for left, right in zip(
                self.cumulative_counts,
                self.cumulative_counts[1:],
                strict=False,
            )
        ):
            raise ValueError("histogram counts must be cumulative")
        if self.cumulative_counts and self.cumulative_counts[-1] > self.count:
            raise ValueError("finite histogram buckets must not exceed the total count")
        if self.count < 0 or not isfinite(self.total) or self.total < 0:
            raise ValueError("histogram count and total must be non-negative")


@dataclass(frozen=True, slots=True)
class StepObservation:
    """一个已经完成的设备步骤及其调度规模。"""

    num_scheduled_tokens: int
    num_requests: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if type(self.num_scheduled_tokens) is not int or self.num_scheduled_tokens <= 0:
            raise ValueError("num_scheduled_tokens must be a positive integer")
        if type(self.num_requests) is not int or self.num_requests <= 0:
            raise ValueError("num_requests must be a positive integer")
        if not isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class StepLatencySnapshot:
    """同一 scheduled-token 桶内的 step 延迟分布。"""

    max_scheduled_tokens: int | None
    latency: HistogramSnapshot

    def __post_init__(self) -> None:
        if self.max_scheduled_tokens is not None and (
            type(self.max_scheduled_tokens) is not int or self.max_scheduled_tokens <= 0
        ):
            raise ValueError("max_scheduled_tokens must be positive or None")


@dataclass(frozen=True, slots=True)
class PerformanceSnapshot:
    """Grafana、Prometheus 或日志适配器可以读取的不可变性能快照。"""

    model_name: str
    scheduler: SchedulerStats
    time_to_first_token: HistogramSnapshot
    inter_token_latency: HistogramSnapshot
    step_latency: tuple[StepLatencySnapshot, ...]
    prompt_tokens_total: int
    generation_tokens_total: int
    finished_requests_total: int
    failed_requests_total: int
    cancelled_requests_total: int
    rejected_requests_total: int
    overloaded_requests_total: int

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        step_latency = tuple(self.step_latency)
        if any(
            type(value) is not int or value < 0
            for value in (
                self.prompt_tokens_total,
                self.generation_tokens_total,
                self.finished_requests_total,
                self.failed_requests_total,
                self.cancelled_requests_total,
                self.rejected_requests_total,
                self.overloaded_requests_total,
            )
        ):
            raise ValueError("performance counters must be non-negative integers")
        object.__setattr__(self, "step_latency", step_latency)


class PerformanceMetricsReader(Protocol):
    """控制面读取性能事实的稳定端口。"""

    def snapshot(self) -> PerformanceSnapshot: ...


class PerformanceObserver(Protocol):
    """Engine 在状态已经生效后调用的轻量观察者。

    回调只更新进程内计数，不执行 I/O，也不得反向修改 Engine、Scheduler
    或 KV cache。具体 Prometheus/Grafana 表达留在 serving adapter。
    """

    def request_started(self, request_id: str, *, num_prompt_tokens: int) -> None: ...

    def request_rejected(self, *, reason: AdmissionRejection) -> None: ...

    def tokens_generated(self, request_id: str, *, count: int) -> None: ...

    def request_finished(self, request_id: str, *, outcome: RequestOutcome) -> None: ...

    def scheduler_updated(self, stats: SchedulerStats) -> None: ...

    def step_completed(self, observation: StepObservation) -> None: ...
