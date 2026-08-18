from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from itertools import accumulate
from threading import RLock
from time import perf_counter

from light_vllm.runtime.kv_cache import KVCacheStats
from light_vllm.runtime.observability.interfaces import (
    AdmissionRejection,
    HistogramSnapshot,
    PerformanceSnapshot,
    RequestOutcome,
    StepLatencySnapshot,
    StepObservation,
)
from light_vllm.runtime.scheduler.interfaces import SchedulerStats

_LATENCY_BOUNDS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
)

_STEP_TOKEN_BOUNDS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


class _Histogram:
    """热路径只更新一个桶，读取快照时再计算累计值。"""

    def __init__(self, bounds: tuple[float, ...]) -> None:
        self._bounds = bounds
        # 最后一个元素保存 +Inf 桶独有的样本数。
        self._bucket_counts = [0] * (len(bounds) + 1)
        self._count = 0
        self._total = 0.0

    def observe(self, value: float, *, count: int = 1) -> None:
        if value < 0:
            raise ValueError("latency observation must be non-negative")
        if type(count) is not int or count <= 0:
            raise ValueError("histogram observation count must be positive")
        bucket = bisect_left(self._bounds, value)
        self._bucket_counts[bucket] += count
        self._count += count
        self._total += value * count

    def snapshot(self) -> HistogramSnapshot:
        return HistogramSnapshot(
            bounds=self._bounds,
            cumulative_counts=tuple(accumulate(self._bucket_counts[:-1])),
            count=self._count,
            total=self._total,
        )


@dataclass(slots=True)
class _RequestTiming:
    started_at: float
    last_token_at: float | None = None


class InMemoryPerformanceObserver:
    """记录 Engine 控制面性能事实，不依赖模型类型或监控后端。"""

    def __init__(
        self,
        model_name: str,
        *,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        self._model_name = model_name
        self._clock = clock
        self._lock = RLock()
        self._requests: dict[str, _RequestTiming] = {}
        self._scheduler = SchedulerStats(
            waiting_requests=0,
            running_requests=0,
            waiting_pending_tokens=0,
            running_pending_tokens=0,
            waiting_max_remaining_tokens=0,
            running_max_remaining_tokens=0,
            kv_cache=KVCacheStats(),
        )
        self._ttft = _Histogram(_LATENCY_BOUNDS)
        self._inter_token_latency = _Histogram(_LATENCY_BOUNDS)
        self._step_latency: dict[int | None, _Histogram] = {}
        self._prompt_tokens_total = 0
        self._generation_tokens_total = 0
        self._request_outcomes: dict[RequestOutcome, int] = {
            "finished": 0,
            "failed": 0,
            "cancelled": 0,
        }
        self._admission_rejections: dict[AdmissionRejection, int] = {
            "capacity": 0,
            "overloaded": 0,
        }

    def request_started(self, request_id: str, *, num_prompt_tokens: int) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if type(num_prompt_tokens) is not int or num_prompt_tokens <= 0:
            raise ValueError("num_prompt_tokens must be a positive integer")
        with self._lock:
            if request_id in self._requests:
                raise ValueError(f"request {request_id!r} is already observed")
            self._requests[request_id] = _RequestTiming(
                started_at=self._clock(),
            )
            # 统计所有已经进入 Engine 的 prompt，包括随后失败或取消的请求。
            self._prompt_tokens_total += num_prompt_tokens

    def request_rejected(self, *, reason: AdmissionRejection) -> None:
        with self._lock:
            self._admission_rejections[reason] += 1

    def tokens_generated(self, request_id: str, *, count: int) -> None:
        if type(count) is not int or count <= 0:
            raise ValueError("generated token count must be a positive integer")
        with self._lock:
            timing = self._request(request_id)
            now = self._clock()
            if timing.last_token_at is None:
                self._ttft.observe(now - timing.started_at)
                # 同一次执行返回多个确认 token 时，它们在同一安全点可见。
                if count > 1:
                    self._inter_token_latency.observe(0.0, count=count - 1)
            else:
                # 一批 token 在同一安全点可见：第一个跨越完整时间间隔，其余间隔为零。
                self._inter_token_latency.observe(now - timing.last_token_at)
                if count > 1:
                    self._inter_token_latency.observe(0.0, count=count - 1)
            timing.last_token_at = now
            self._generation_tokens_total += count

    def request_finished(self, request_id: str, *, outcome: RequestOutcome) -> None:
        with self._lock:
            self._request(request_id)
            del self._requests[request_id]
            self._request_outcomes[outcome] += 1

    def scheduler_updated(self, stats: SchedulerStats) -> None:
        with self._lock:
            self._scheduler = stats

    def step_completed(self, observation: StepObservation) -> None:
        if not isinstance(observation, StepObservation):
            raise TypeError("observation must be StepObservation")
        bucket = next(
            (bound for bound in _STEP_TOKEN_BOUNDS if observation.num_scheduled_tokens <= bound),
            None,
        )
        with self._lock:
            histogram = self._step_latency.setdefault(bucket, _Histogram(_LATENCY_BOUNDS))
            histogram.observe(observation.elapsed_seconds)

    def snapshot(self) -> PerformanceSnapshot:
        with self._lock:
            return PerformanceSnapshot(
                model_name=self._model_name,
                scheduler=self._scheduler,
                time_to_first_token=self._ttft.snapshot(),
                inter_token_latency=self._inter_token_latency.snapshot(),
                step_latency=tuple(
                    StepLatencySnapshot(
                        max_scheduled_tokens=bucket,
                        latency=histogram.snapshot(),
                    )
                    for bucket, histogram in sorted(
                        self._step_latency.items(),
                        key=lambda item: (
                            item[0] is None,
                            item[0] if item[0] is not None else 0,
                        ),
                    )
                ),
                prompt_tokens_total=self._prompt_tokens_total,
                generation_tokens_total=self._generation_tokens_total,
                finished_requests_total=self._request_outcomes["finished"],
                failed_requests_total=self._request_outcomes["failed"],
                cancelled_requests_total=self._request_outcomes["cancelled"],
                rejected_requests_total=self._admission_rejections["capacity"],
                overloaded_requests_total=self._admission_rejections["overloaded"],
            )

    def _request(self, request_id: str) -> _RequestTiming:
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise ValueError(f"request {request_id!r} is not observed") from exc
