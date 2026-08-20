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


def _scheduler_stats(
    *,
    waiting_tokens: int,
    running_tokens: int,
    waiting_requests: int = 1,
    running_requests: int = 1,
    kv_cache: KVCacheStats | None = None,
) -> SchedulerStats:
    return SchedulerStats(
        waiting_requests=waiting_requests,
        running_requests=running_requests,
        waiting_pending_tokens=waiting_tokens,
        running_pending_tokens=running_tokens,
        waiting_max_remaining_tokens=waiting_tokens,
        running_max_remaining_tokens=running_tokens,
        kv_cache=kv_cache or KVCacheStats(),
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


def test_request_ttft_slo_works_without_a_global_slo() -> None:
    predictor = SlidingWindowStepLatencyPredictor(min_observations=1)
    _observe(predictor, 9, 0.4)
    admission = PredictiveTTFTAdmission(predictor)

    with pytest.raises(GenerationOverloadedError, match="the 0.300s SLO"):
        admission.validate(
            GenerateRequest(
                input_ids=(1, 2),
                max_new_tokens=8,
                max_tolerable_ttft_seconds=0.3,
            ),
            _scheduler_stats(waiting_tokens=4, running_tokens=3),
        )


def test_pending_request_gate_applies_before_predictor_warmup() -> None:
    predictor = SlidingWindowStepLatencyPredictor(min_observations=100)
    admission = PredictiveTTFTAdmission(predictor, max_pending_requests=4)

    with pytest.raises(GenerationOverloadedError, match="pending request limit 4"):
        admission.validate(
            GenerateRequest(input_ids=(1,), max_new_tokens=1),
            _scheduler_stats(
                waiting_tokens=3,
                running_tokens=1,
                waiting_requests=3,
                running_requests=1,
            ),
        )


def test_kv_gate_counts_used_and_claimed_capacity() -> None:
    admission = PredictiveTTFTAdmission(
        SlidingWindowStepLatencyPredictor(min_observations=100),
        kv_cache_watermark=0.9,
    )

    with pytest.raises(GenerationOverloadedError, match="KV cache watermark 0.900"):
        admission.validate(
            GenerateRequest(input_ids=(1,), max_new_tokens=1),
            _scheduler_stats(
                waiting_tokens=1,
                running_tokens=1,
                kv_cache=KVCacheStats(
                    used_token_slots=80,
                    claimed_token_slots=10,
                    capacity_token_slots=100,
                ),
            ),
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_tolerable_ttft_seconds", 0),
        ("max_tolerable_ttft_seconds", True),
        ("max_pending_requests", 0),
        ("kv_cache_watermark", 0),
        ("kv_cache_watermark", 1.1),
    ],
)
def test_predictive_ttft_admission_rejects_invalid_gates(name: str, value: object) -> None:
    with pytest.raises(ValueError):
        PredictiveTTFTAdmission(
            SlidingWindowStepLatencyPredictor(),
            **{name: value},
        )


def test_predictor_failure_does_not_disable_queue_and_kv_gates() -> None:
    class FailingPredictor:
        def predict(self, num_pending_tokens: int) -> float | None:
            raise RuntimeError("predictor failed")

        def observe(self, observation: StepObservation) -> None:
            raise RuntimeError("predictor failed")

    admission = PredictiveTTFTAdmission(
        FailingPredictor(),
        max_tolerable_ttft_seconds=0.5,
        max_pending_requests=2,
    )
    admission.validate(
        GenerateRequest(input_ids=(1,), max_new_tokens=1),
        _scheduler_stats(
            waiting_tokens=0,
            running_tokens=1,
            waiting_requests=0,
            running_requests=1,
        ),
    )

    with pytest.raises(GenerationOverloadedError, match="pending request limit 2"):
        admission.validate(
            GenerateRequest(input_ids=(1,), max_new_tokens=1),
            _scheduler_stats(
                waiting_tokens=1,
                running_tokens=1,
                waiting_requests=1,
                running_requests=1,
            ),
        )
