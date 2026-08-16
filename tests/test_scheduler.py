import pytest

from light_vllm.runtime.kv_cache import (
    FixedKVBlockCapacity,
    KVCacheMatch,
    PagedKVCacheManager,
    UnboundedKVCacheManager,
)
from light_vllm.runtime.scheduler import DecodingBudget, SchedulerError, TokenBudgetScheduler


def _scheduler(*, token_budget: int = 4, num_blocks: int = 8) -> TokenBudgetScheduler:
    return TokenBudgetScheduler(
        PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=num_blocks, block_size=2)),
        max_num_sequences=2,
        max_num_scheduled_tokens=token_budget,
    )


def test_scheduler_chunks_long_prompts_with_one_token_budget() -> None:
    scheduler = _scheduler(token_budget=2)
    scheduler.add("request", token_ids=(1, 2, 3, 4, 5))

    first = scheduler.schedule().requests[0]
    assert first.num_computed_tokens == 0
    assert first.num_scheduled_tokens == 2
    assert first.max_output_tokens == 0
    scheduler.complete("request", num_committed_tokens=2, num_new_tokens=0)

    second = scheduler.schedule().requests[0]
    assert second.num_computed_tokens == 2
    assert second.num_scheduled_tokens == 2
    assert second.max_output_tokens == 0
    scheduler.complete("request", num_committed_tokens=2, num_new_tokens=0)

    last = scheduler.schedule().requests[0]
    assert last.num_computed_tokens == 4
    assert last.num_scheduled_tokens == 1
    assert last.max_output_tokens == 1


def test_scheduler_uses_one_budget_across_requests_and_refills_open_slots() -> None:
    scheduler = _scheduler(token_budget=3)
    scheduler.add("a", token_ids=(1, 2))
    scheduler.add("b", token_ids=(3, 4))
    scheduler.add("c", token_ids=(5,))

    output = scheduler.schedule()
    assert [(item.request_id, item.num_scheduled_tokens) for item in output.requests] == [
        ("a", 2),
        ("b", 1),
    ]
    scheduler.complete("b", num_committed_tokens=1, num_new_tokens=0)
    assert scheduler.remove("a")

    output = scheduler.schedule()
    assert output.request_ids == ("b", "c")


def test_scheduler_rejects_duplicate_request_ids() -> None:
    scheduler = _scheduler()
    scheduler.add("request", token_ids=(1,))

    with pytest.raises(SchedulerError, match="already scheduled"):
        scheduler.add("request", token_ids=(1,))


def test_scheduler_supports_reservations_without_block_placement() -> None:
    scheduler = TokenBudgetScheduler(
        UnboundedKVCacheManager(),
        max_num_sequences=1,
        max_num_scheduled_tokens=2,
    )
    scheduler.add("request", token_ids=(1, 2, 3))

    first = scheduler.schedule().requests[0]
    assert first.block_ids is None
    assert first.num_scheduled_tokens == 2
    scheduler.complete("request", num_committed_tokens=2, num_new_tokens=0)

    second = scheduler.schedule().requests[0]
    assert second.block_ids is None
    assert second.num_computed_tokens == 2


def test_scheduler_reserves_lookahead_without_a_decode_mode() -> None:
    scheduler = TokenBudgetScheduler(
        PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=4, block_size=1)),
        max_num_sequences=1,
        max_num_scheduled_tokens=2,
        decoding_budget=DecodingBudget(
            num_lookahead_tokens=1,
            max_output_tokens=2,
        ),
    )
    scheduler.add("request", token_ids=(1,))

    first = scheduler.schedule().requests[0]
    assert first.num_scheduled_tokens == 1
    assert first.num_lookahead_tokens == 1
    assert first.max_output_tokens == 2

    # 一个输入和一个已接受输出已经写入 KV，bonus token 留到下一轮计算。
    scheduler.complete("request", num_committed_tokens=2, num_new_tokens=2)
    second = scheduler.schedule().requests[0]
    assert second.num_computed_tokens == 2
    assert second.num_scheduled_tokens == 1


def test_scheduler_releases_an_invalid_prefix_match() -> None:
    class InvalidMatchCache:
        def __init__(self) -> None:
            self.freed: list[str] = []

        def add_request(self, request_id, *, token_ids, cache_epoch):
            return KVCacheMatch(num_cached_tokens=len(token_ids))

        def reserve(self, request_id, num_tokens):
            raise AssertionError("invalid prefix must fail before reservation")

        def commit(self, request_id, num_tokens):
            raise AssertionError("invalid prefix must fail before commit")

        def free(self, request_id):
            self.freed.append(request_id)
            return True

    cache = InvalidMatchCache()
    scheduler = TokenBudgetScheduler(
        cache,
        max_num_sequences=1,
        max_num_scheduled_tokens=2,
    )
    scheduler.add("request", token_ids=(1, 2))

    with pytest.raises(SchedulerError, match="leave at least one token"):
        scheduler.schedule()
    assert cache.freed == ["request"]
