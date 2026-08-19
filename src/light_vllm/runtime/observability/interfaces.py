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
    """一个已经完成的设备步骤及其真实模型计算规模。"""

    num_model_tokens_computed: int
    num_requests: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if type(self.num_model_tokens_computed) is not int or self.num_model_tokens_computed <= 0:
            raise ValueError("num_model_tokens_computed must be a positive integer")
        if type(self.num_requests) is not int or self.num_requests <= 0:
            raise ValueError("num_requests must be a positive integer")
        if not isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class StepLatencySnapshot:
    """同一实际模型 token 桶内的 step 延迟分布。"""

    max_model_tokens_computed: int | None
    latency: HistogramSnapshot

    def __post_init__(self) -> None:
        if self.max_model_tokens_computed is not None and (
            type(self.max_model_tokens_computed) is not int or self.max_model_tokens_computed <= 0
        ):
            raise ValueError("max_model_tokens_computed must be positive or None")


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
    speculation_attempts_total: int
    speculation_hits_total: int
    speculative_proposed_nodes_total: int
    speculative_accepted_nodes_total: int
    speculative_verified_tokens_total: int
    speculative_draft_roots_total: int
    speculative_branching_parents_total: int
    speculative_compacted_tokens_total: int
    speculative_max_draft_depth: int

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
                self.speculation_attempts_total,
                self.speculation_hits_total,
                self.speculative_proposed_nodes_total,
                self.speculative_accepted_nodes_total,
                self.speculative_verified_tokens_total,
                self.speculative_draft_roots_total,
                self.speculative_branching_parents_total,
                self.speculative_compacted_tokens_total,
                self.speculative_max_draft_depth,
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
