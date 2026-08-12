import pytest

from light_vllm.scheduler import (
    ContinuousBatchScheduler,
    SchedulerError,
    StaticBatchScheduler,
)


def test_continuous_batching_refills_open_slots_each_iteration() -> None:
    scheduler = ContinuousBatchScheduler(max_num_sequences=2)
    for request_id in ("a", "b", "c"):
        scheduler.add(request_id)

    assert scheduler.schedule().request_ids == ("a", "b")
    assert scheduler.remove("a")
    assert scheduler.schedule().request_ids == ("b", "c")


def test_static_batching_waits_until_the_current_wave_is_empty() -> None:
    scheduler = StaticBatchScheduler(max_num_sequences=2)
    for request_id in ("a", "b", "c"):
        scheduler.add(request_id)

    assert scheduler.schedule().request_ids == ("a", "b")
    assert scheduler.remove("a")
    assert scheduler.schedule().request_ids == ("b",)
    assert scheduler.remove("b")
    assert scheduler.schedule().request_ids == ("c",)


def test_scheduler_rejects_duplicate_request_ids() -> None:
    scheduler = ContinuousBatchScheduler(max_num_sequences=1)
    scheduler.add("request")

    with pytest.raises(SchedulerError, match="already scheduled"):
        scheduler.add("request")
