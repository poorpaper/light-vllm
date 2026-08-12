import pytest

from light_vllm.runtime.kv_cache import PagedKVCacheManager
from light_vllm.runtime.scheduler import SchedulerError, TokenBudgetScheduler


def _scheduler(*, token_budget: int = 4, num_blocks: int = 8) -> TokenBudgetScheduler:
    return TokenBudgetScheduler(
        PagedKVCacheManager(num_blocks=num_blocks, block_size=2),
        max_num_sequences=2,
        max_num_scheduled_tokens=token_budget,
    )


def test_scheduler_chunks_long_prompts_with_one_token_budget() -> None:
    scheduler = _scheduler(token_budget=2)
    scheduler.add("request", num_tokens=5)

    first = scheduler.schedule().requests[0]
    assert first.num_computed_tokens == 0
    assert first.num_scheduled_tokens == 2
    assert not first.sampling_required
    scheduler.complete("request", num_computed_tokens=2, num_new_tokens=0)

    second = scheduler.schedule().requests[0]
    assert second.num_computed_tokens == 2
    assert second.num_scheduled_tokens == 2
    assert not second.sampling_required
    scheduler.complete("request", num_computed_tokens=2, num_new_tokens=0)

    last = scheduler.schedule().requests[0]
    assert last.num_computed_tokens == 4
    assert last.num_scheduled_tokens == 1
    assert last.sampling_required


def test_scheduler_uses_one_budget_across_requests_and_refills_open_slots() -> None:
    scheduler = _scheduler(token_budget=3)
    scheduler.add("a", num_tokens=2)
    scheduler.add("b", num_tokens=2)
    scheduler.add("c", num_tokens=1)

    output = scheduler.schedule()
    assert [(item.request_id, item.num_scheduled_tokens) for item in output.requests] == [
        ("a", 2),
        ("b", 1),
    ]
    scheduler.complete("b", num_computed_tokens=1, num_new_tokens=0)
    assert scheduler.remove("a")

    output = scheduler.schedule()
    assert output.request_ids == ("b", "c")


def test_scheduler_rejects_duplicate_request_ids() -> None:
    scheduler = _scheduler()
    scheduler.add("request", num_tokens=1)

    with pytest.raises(SchedulerError, match="already scheduled"):
        scheduler.add("request", num_tokens=1)
