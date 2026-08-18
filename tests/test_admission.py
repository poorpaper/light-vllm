import pytest

from light_vllm.runtime.engine.admission import (
    PredictiveTTFTAdmission,
    SlidingWindowStepLatencyPredictor,
)
from light_vllm.runtime.generation import GenerateRequest, GenerationOverloadedError
from light_vllm.runtime.kv_cache import KVCacheStats
from light_vllm.runtime.observability import StepObservation
from light_vllm.runtime.scheduler import SchedulerStats


def _observe(
    predictor: SlidingWindowStepLatencyPredictor,
    tokens: int,
    latency: float,
) -> None:
    predictor.observe(
        StepObservation(
            num_model_tokens_computed=tokens,
            num_requests=1,
            elapsed_seconds=latency,
        )
    )


def _scheduler_stats(*, waiting_tokens: int, running_tokens: int) -> SchedulerStats:
    return SchedulerStats(
        waiting_requests=1,
        running_requests=1,
        waiting_pending_tokens=waiting_tokens,
        running_pending_tokens=running_tokens,
        waiting_max_remaining_tokens=waiting_tokens,
        running_max_remaining_tokens=running_tokens,
        kv_cache=KVCacheStats(),
    )


def test_step_latency_predictor_uses_quantile_interpolation_and_extrapolation() -> None:
    predictor = SlidingWindowStepLatencyPredictor(window_size=2, min_observations=1)
    for latency in (0.02, 0.04, 0.06):
        _observe(predictor, 2, latency)
    _observe(predictor, 6, 0.12)

    assert predictor.predict(1) == pytest.approx(0.06)
    assert predictor.predict(2) == pytest.approx(0.06)
    assert predictor.predict(4) == pytest.approx(0.09)
    assert predictor.predict(12) == pytest.approx(0.24)


def test_step_latency_predictor_is_fail_open_until_warm() -> None:
    predictor = SlidingWindowStepLatencyPredictor(min_observations=2)

    _observe(predictor, 4, 0.1)
    assert predictor.predict(4) is None
    _observe(predictor, 4, 0.2)

    assert predictor.predict(4) == pytest.approx(0.2)


@pytest.mark.parametrize("quantile", (0.0, -0.1, 1.1, float("inf")))
def test_step_latency_predictor_rejects_invalid_quantiles(quantile: float) -> None:
    with pytest.raises(ValueError, match="prediction_quantile"):
        SlidingWindowStepLatencyPredictor(prediction_quantile=quantile)


def test_step_latency_predictor_keeps_slow_steps_and_token_sizes_conservative() -> None:
    predictor = SlidingWindowStepLatencyPredictor(
        window_size=5,
        min_observations=1,
        prediction_quantile=0.9,
    )
    for latency in (0.02, 0.02, 0.02, 0.02, 0.2):
        _observe(predictor, 2, latency)
    _observe(predictor, 6, 0.1)

    # p90 保留窗口里的慢 step，单调包络不让更大负载得到更低预测。
    assert predictor.predict(2) == pytest.approx(0.2)
    assert predictor.predict(4) == pytest.approx(0.2)
    assert predictor.predict(6) == pytest.approx(0.2)


def test_predictive_ttft_admission_uses_prompt_plus_current_pending_tokens() -> None:
    predictor = SlidingWindowStepLatencyPredictor(min_observations=1)
    _observe(predictor, 9, 0.6)
    admission = PredictiveTTFTAdmission(
        predictor,
        max_tolerable_ttft_seconds=0.5,
    )

    with pytest.raises(GenerationOverloadedError, match="predicted TTFT 0.600s"):
        admission.validate(
            GenerateRequest(input_ids=(1, 2), max_new_tokens=8),
            _scheduler_stats(waiting_tokens=4, running_tokens=3),
        )


def test_predictive_ttft_admission_uses_global_current_work_not_future_outputs() -> None:
    class RecordingPredictor:
        def __init__(self) -> None:
            self.inputs: list[int] = []

        def predict(self, num_pending_tokens: int) -> None:
            self.inputs.append(num_pending_tokens)
            return None

        def observe(self, observation: StepObservation) -> None:
            return

    predictor = RecordingPredictor()
    admission = PredictiveTTFTAdmission(predictor, max_tolerable_ttft_seconds=0.5)
    stats = SchedulerStats(
        waiting_requests=3,
        running_requests=300,
        waiting_pending_tokens=8_000,
        running_pending_tokens=300,
        waiting_max_remaining_tokens=50_000,
        running_max_remaining_tokens=100_000,
        kv_cache=KVCacheStats(),
    )

    admission.validate(GenerateRequest(input_ids=(1, 2), max_new_tokens=4096), stats)

    # 300 个 decode 各贡献当前 1 token，再与剩余 prefill 和新 prompt 全局求和。
    assert stats.current_pending_tokens == 8_300
    assert predictor.inputs == [8_302]


def test_predictive_ttft_admission_does_not_average_away_a_slow_step() -> None:
    predictor = SlidingWindowStepLatencyPredictor(
        window_size=5,
        min_observations=5,
        prediction_quantile=0.9,
    )
    for latency in (0.1, 0.1, 0.1, 0.1, 0.6):
        _observe(predictor, 9, latency)
    admission = PredictiveTTFTAdmission(
        predictor,
        max_tolerable_ttft_seconds=0.5,
    )

    with pytest.raises(GenerationOverloadedError, match="predicted TTFT 0.600s"):
        admission.validate(
            GenerateRequest(input_ids=(1, 2), max_new_tokens=8),
            _scheduler_stats(waiting_tokens=4, running_tokens=3),
        )
