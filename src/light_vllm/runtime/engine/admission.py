"""请求进入调度队列前的容量与延迟检查。"""

from bisect import bisect_left
from collections import deque
from math import ceil, isfinite
from threading import RLock

from light_vllm.runtime.engine.interfaces import (
    EngineCapabilities,
    StepLatencyPredictor,
    TTFTAdmission,
)
from light_vllm.runtime.generation.interfaces import (
    GenerateRequest,
    GenerationOverloadedError,
    GenerationRejectedError,
)
from light_vllm.runtime.observability.interfaces import StepObservation
from light_vllm.runtime.scheduler.interfaces import SchedulerStats


class CapacityAdmission:
    """只拒绝即使独占整个引擎也装不下的请求。"""

    def validate(self, request: GenerateRequest, capabilities: EngineCapabilities) -> None:
        # 这里只检查请求本身是否过大；当前是否有空闲资源由 Scheduler 决定。
        limit = capabilities.max_request_tokens
        requested_tokens = len(request.input_ids) + request.max_new_tokens
        if limit is not None and requested_tokens > limit:
            raise GenerationRejectedError(
                f"request needs {requested_tokens} tokens, but the engine supports at most {limit}"
            )


class SlidingWindowStepLatencyPredictor:
    """用 scheduled-token 表和滑动窗口保守估计负载延迟。

    每个桶使用上分位数，再构造随 token 数不下降的包络。未见过的
    中间负载做线性插值；超过最大样本时按 token 比例外推。样本不足
    时返回 ``None``，让冷启动阶段保持 fail-open。
    """

    def __init__(
        self,
        *,
        window_size: int = 32,
        min_observations: int = 3,
        prediction_quantile: float = 0.9,
    ) -> None:
        if type(window_size) is not int or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if type(min_observations) is not int or min_observations <= 0:
            raise ValueError("min_observations must be a positive integer")
        if not isfinite(prediction_quantile) or not 0 < prediction_quantile <= 1:
            raise ValueError("prediction_quantile must be in (0, 1]")
        self._window_size = window_size
        self._min_observations = min_observations
        self._prediction_quantile = prediction_quantile
        self._samples: dict[int, deque[float]] = {}
        self._lock = RLock()

    def observe(self, observation: StepObservation) -> None:
        if not isinstance(observation, StepObservation):
            raise TypeError("observation must be StepObservation")
        with self._lock:
            samples = self._samples.setdefault(
                observation.num_scheduled_tokens,
                deque(maxlen=self._window_size),
            )
            samples.append(observation.elapsed_seconds)

    def predict(self, num_pending_tokens: int) -> float | None:
        if type(num_pending_tokens) is not int or num_pending_tokens <= 0:
            raise ValueError("num_pending_tokens must be a positive integer")
        with self._lock:
            if sum(len(samples) for samples in self._samples.values()) < self._min_observations:
                return None
            table: list[tuple[int, float]] = []
            monotonic_latency = 0.0
            for tokens, samples in sorted(self._samples.items()):
                ordered = sorted(samples)
                rank = ceil(self._prediction_quantile * len(ordered)) - 1
                monotonic_latency = max(monotonic_latency, ordered[rank])
                table.append((tokens, monotonic_latency))

        token_sizes = tuple(tokens for tokens, _ in table)
        index = bisect_left(token_sizes, num_pending_tokens)
        if index == 0:
            return table[0][1]
        if index == len(table):
            largest_tokens, largest_latency = table[-1]
            return largest_latency * num_pending_tokens / largest_tokens

        left_tokens, left_latency = table[index - 1]
        right_tokens, right_latency = table[index]
        ratio = (num_pending_tokens - left_tokens) / (right_tokens - left_tokens)
        return left_latency + ratio * (right_latency - left_latency)


class PredictiveTTFTAdmission:
    """请求入队前，用当前 pending token 总量检查 TTFT SLO。"""

    def __init__(
        self,
        predictor: StepLatencyPredictor,
        *,
        max_tolerable_ttft_seconds: float,
    ) -> None:
        if not isfinite(max_tolerable_ttft_seconds) or max_tolerable_ttft_seconds <= 0:
            raise ValueError("max_tolerable_ttft_seconds must be finite and positive")
        self._predictor = predictor
        self._max_tolerable_ttft_seconds = max_tolerable_ttft_seconds

    def validate(self, request: GenerateRequest, stats: SchedulerStats) -> None:
        pending_tokens = (
            len(request.input_ids) + stats.waiting_pending_tokens + stats.running_pending_tokens
        )
        prediction = self._predictor.predict(pending_tokens)
        if prediction is not None and prediction > self._max_tolerable_ttft_seconds:
            raise GenerationOverloadedError(
                f"predicted TTFT {prediction:.3f}s exceeds "
                f"the {self._max_tolerable_ttft_seconds:.3f}s SLO"
            )

    def step_completed(self, observation: StepObservation) -> None:
        self._predictor.observe(observation)


class SafeTTFTAdmission:
    """预测器故障时禁用动态早拒，不让控制面故障打断生成。"""

    def __init__(self, admission: TTFTAdmission | None) -> None:
        self._admission = admission
        self._lock = RLock()

    def validate(self, request: GenerateRequest, stats: SchedulerStats) -> None:
        with self._lock:
            admission = self._admission
            if admission is None:
                return
            try:
                admission.validate(request, stats)
            except GenerationOverloadedError:
                raise
            except Exception:
                self._admission = None

    def step_completed(self, observation: StepObservation) -> None:
        with self._lock:
            admission = self._admission
            if admission is None:
                return
            try:
                admission.step_completed(observation)
            except Exception:
                self._admission = None
