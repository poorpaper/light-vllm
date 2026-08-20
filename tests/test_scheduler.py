import pytest

from light_vllm.runtime.kv_cache import (
    FixedKVBlockCapacity,
    KVCacheMatch,
    PagedKVCacheManager,
    UnboundedKVCacheManager,
)
from light_vllm.runtime.scheduler import (
    DecodingBudget,
    SchedulerError,
    SelfResubmitPolicy,
    ShortRequestPolicy,
    TokenBudgetScheduler,
)


def _scheduler(*, token_budget: int = 4, num_blocks: int = 8) -> TokenBudgetScheduler:
    return TokenBudgetScheduler(
        PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=num_blocks, block_size=2)),
        max_num_sequences=2,
        max_num_scheduled_tokens=token_budget,
    )


def _short_policy(
    *,
    max_effective_prompt_tokens: int = 2,
    max_total_tokens: int = 4,
    reserved_scheduled_tokens: int = 2,
    reserved_kv_token_slots: int = 3,
    reserved_sequences: int = 1,
    regular_aging_steps: int = 3,
) -> ShortRequestPolicy:
    return ShortRequestPolicy(
        max_effective_prompt_tokens=max_effective_prompt_tokens,
        max_total_tokens=max_total_tokens,
        reserved_scheduled_tokens=reserved_scheduled_tokens,
        reserved_kv_token_slots=reserved_kv_token_slots,
        reserved_sequences=reserved_sequences,
        regular_aging_steps=regular_aging_steps,
    )


def test_scheduler_chunks_long_prompts_with_one_token_budget() -> None:
    scheduler = _scheduler(token_budget=2)
    scheduler.add("request", token_ids=(1, 2, 3, 4, 5), max_num_tokens=9)

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
    scheduler.add("a", token_ids=(1, 2), max_num_tokens=6)
    scheduler.add("b", token_ids=(3, 4), max_num_tokens=6)
    scheduler.add("c", token_ids=(5,), max_num_tokens=5)

    output = scheduler.schedule()
    assert [(item.request_id, item.num_scheduled_tokens) for item in output.requests] == [
        ("a", 2),
        ("b", 1),
    ]
    scheduler.complete("b", num_committed_tokens=1, num_new_tokens=0)
    assert scheduler.remove("a")

    output = scheduler.schedule()
    assert output.request_ids == ("b", "c")


def test_scheduler_keeps_unsafe_completion_claims_waiting() -> None:
    scheduler = TokenBudgetScheduler(
        PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=6, block_size=1)),
        max_num_sequences=2,
        max_num_scheduled_tokens=2,
    )
    scheduler.add("a", token_ids=(1,), max_num_tokens=5)
    scheduler.add("b", token_ids=(2,), max_num_tokens=4)

    first = scheduler.schedule()

    assert first.request_ids == ("a",)
    assert scheduler.stats.running_requests == 1
    assert scheduler.stats.waiting_requests == 1
    assert scheduler.stats.kv_cache.used_token_slots == 1
    assert scheduler.stats.kv_cache.claimed_token_slots == 3

    scheduler.remove("a")
    second = scheduler.schedule()
    assert second.request_ids == ("b",)


def test_short_request_reserve_runs_beside_a_long_prefill() -> None:
    scheduler = TokenBudgetScheduler(
        PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=16, block_size=1)),
        max_num_sequences=2,
        max_num_scheduled_tokens=4,
        short_request_policy=_short_policy(),
    )
    scheduler.add("long", token_ids=(1, 2, 3, 4), max_num_tokens=8)

    first = scheduler.schedule().requests[0]
    assert (first.request_id, first.num_scheduled_tokens) == ("long", 2)
    scheduler.complete("long", num_committed_tokens=2, num_new_tokens=0)
    scheduler.add("short", token_ids=(5, 6), max_num_tokens=4)

    output = scheduler.schedule()

    assert scheduler.stats.short_launch_requests == 1
    assert [
        (item.request_id, item.num_scheduled_tokens, item.max_output_tokens)
        for item in output.requests
    ] == [("short", 2, 1), ("long", 2, 1)]

    for item in output.requests:
        scheduler.complete(
            item.request_id,
            num_committed_tokens=item.num_scheduled_tokens,
            num_new_tokens=1,
        )
    assert scheduler.stats.short_launch_requests == 0

    # 首 token 之后短请求只能进入通用池，不能继续占用预留池。
    assert scheduler.schedule().request_ids == ("long",)


def test_short_kv_headroom_is_not_claimed_by_regular_requests() -> None:
    scheduler = TokenBudgetScheduler(
        PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=7, block_size=1)),
        max_num_sequences=2,
        max_num_scheduled_tokens=4,
        short_request_policy=_short_policy(),
    )
    scheduler.add("regular", token_ids=(1,), max_num_tokens=6)
    scheduler.add("short", token_ids=(2, 3), max_num_tokens=4)

    output = scheduler.schedule()

    assert output.request_ids == ("short",)
    assert scheduler.stats.running_requests == 1
    assert scheduler.stats.waiting_requests == 1


def test_cached_prefix_uses_effective_prompt_for_short_classification() -> None:
    cache = PagedKVCacheManager(
        FixedKVBlockCapacity(num_blocks=12, block_size=2),
        enable_prefix_caching=True,
    )
    warm_tokens = (1, 2, 3, 4, 5)
    cache.try_add_request(
        "warm",
        token_ids=warm_tokens,
        max_num_committed_tokens=len(warm_tokens),
        cache_epoch=1,
    )
    cache.reserve("warm", len(warm_tokens))
    cache.commit("warm", len(warm_tokens))
    cache.free("warm")
    scheduler = TokenBudgetScheduler(
        cache,
        max_num_sequences=2,
        max_num_scheduled_tokens=3,
        short_request_policy=_short_policy(
            max_effective_prompt_tokens=1,
            max_total_tokens=6,
            reserved_scheduled_tokens=1,
            reserved_kv_token_slots=5,
        ),
    )
    scheduler.add("regular", token_ids=(8, 9), max_num_tokens=7, cache_epoch=1)
    scheduler.add("hit", token_ids=(1, 2, 3, 4, 7), max_num_tokens=6, cache_epoch=1)

    output = scheduler.schedule()

    assert output.request_ids == ("hit", "regular")
    assert output.requests[0].num_computed_tokens == 4
    assert output.requests[0].max_output_tokens == 1


def test_aged_regular_request_can_borrow_short_kv_headroom() -> None:
    scheduler = TokenBudgetScheduler(
        PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=10, block_size=1)),
        max_num_sequences=3,
        max_num_scheduled_tokens=3,
        short_request_policy=_short_policy(
            reserved_scheduled_tokens=2,
            regular_aging_steps=3,
        ),
    )
    scheduler.add("incumbent", token_ids=(1, 2, 3, 4), max_num_tokens=5)
    first = scheduler.schedule().requests[0]
    scheduler.complete(
        "incumbent",
        num_committed_tokens=first.num_scheduled_tokens,
        num_new_tokens=0,
    )
    scheduler.add("aged", token_ids=(5,), max_num_tokens=5)

    for _ in range(2):
        output = scheduler.schedule()
        assert output.request_ids == ("incumbent",)
        scheduler.complete(
            "incumbent",
            num_committed_tokens=output.requests[0].num_scheduled_tokens,
            num_new_tokens=0,
        )

    scheduler.schedule()

    assert scheduler.stats.running_requests == 2
    assert scheduler.stats.waiting_requests == 0


def test_aged_regular_blocks_bulk_short_completion_claims() -> None:
    scheduler = TokenBudgetScheduler(
        PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=23, block_size=1)),
        max_num_sequences=2,
        max_num_scheduled_tokens=4,
        short_request_policy=_short_policy(regular_aging_steps=2),
    )
    scheduler.add("regular", token_ids=(1,), max_num_tokens=20)
    for index in range(10):
        scheduler.add(
            f"short-{index}",
            token_ids=(100 + index, 200 + index),
            max_num_tokens=4,
        )

    first = scheduler.schedule()
    assert first.request_ids == ("short-0",)
    scheduler.complete("short-0", num_committed_tokens=2, num_new_tokens=1)

    second = scheduler.schedule()
    assert second.request_ids == ("short-0",)
    assert scheduler.stats.running_requests == 1
    assert scheduler.stats.waiting_requests == 10
    scheduler.complete("short-0", num_committed_tokens=1, num_new_tokens=1)
    scheduler.remove("short-0")

    third = scheduler.schedule()
    assert third.request_ids == ("short-1", "regular")
    assert scheduler.stats.running_requests == scheduler.max_num_sequences
    assert scheduler.stats.waiting_requests == 8


def test_common_pool_rotates_admitted_requests() -> None:
    scheduler = TokenBudgetScheduler(
        UnboundedKVCacheManager(),
        max_num_sequences=2,
        max_num_scheduled_tokens=1,
    )
    scheduler.add("a", token_ids=(1, 2, 3), max_num_tokens=5)
    scheduler.add("b", token_ids=(4, 5, 6), max_num_tokens=5)

    first = scheduler.schedule()
    scheduler.complete("a", num_committed_tokens=1, num_new_tokens=0)
    second = scheduler.schedule()

    assert first.request_ids == ("a",)
    assert second.request_ids == ("b",)


def test_self_resubmit_recomputes_without_reemitting_and_then_falls_back_to_strict() -> None:
    scheduler = TokenBudgetScheduler(
        PagedKVCacheManager(
            FixedKVBlockCapacity(num_blocks=4, block_size=1),
            enable_prefix_caching=True,
        ),
        max_num_sequences=2,
        max_num_scheduled_tokens=1,
        self_resubmit_policy=SelfResubmitPolicy(
            max_resubmits=1,
            strict_fallback_rolled_back_tokens=100,
            kv_admission_watermark=1.0,
        ),
    )
    scheduler.add("a", token_ids=(1,), max_num_tokens=4, cache_epoch=0)
    scheduler.add("b", token_ids=(2,), max_num_tokens=4, cache_epoch=0)

    for request_id in ("a", "b", "a", "b"):
        output = scheduler.schedule()
        assert output.request_ids == (request_id,)
        scheduler.complete(request_id, num_committed_tokens=1, num_new_tokens=1)

    # a 撞墙后只释放自己的两页；b 使用释放出的空间完成本轮。
    collision = scheduler.schedule()
    assert collision.request_ids == ("b",)
    assert scheduler.stats.waiting_requests == 1
    assert scheduler.stats.self_resubmits_total == 1
    assert scheduler.stats.self_resubmit_rolled_back_tokens_total == 2
    scheduler.complete("b", num_committed_tokens=1, num_new_tokens=1)
    scheduler.remove("b")

    recompute = scheduler.schedule().requests[0]
    assert recompute.request_id == "a"
    assert recompute.num_computed_tokens == 0
    assert recompute.max_output_tokens == 0
    # 已达 resubmit 上限，a 重新准入时恢复了 completion claim。
    assert scheduler.stats.kv_cache.claimed_token_slots == 2


def test_self_resubmit_requires_prefix_caching() -> None:
    with pytest.raises(ValueError, match="self-resubmit requires prefix caching"):
        TokenBudgetScheduler(
            PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=4, block_size=1)),
            max_num_sequences=1,
            max_num_scheduled_tokens=1,
            self_resubmit_policy=SelfResubmitPolicy(),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"initial_extra_blocks": 0},
        {"kv_admission_watermark": 0},
        {"kv_admission_watermark": 1.1},
    ],
)
def test_self_resubmit_policy_rejects_invalid_optimistic_limits(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        SelfResubmitPolicy(**changes)


def test_self_resubmit_does_not_displace_the_reserved_short_lane() -> None:
    scheduler = TokenBudgetScheduler(
        PagedKVCacheManager(
            FixedKVBlockCapacity(num_blocks=5, block_size=1),
            enable_prefix_caching=True,
        ),
        max_num_sequences=3,
        max_num_scheduled_tokens=2,
        short_request_policy=_short_policy(
            max_effective_prompt_tokens=1,
            max_total_tokens=2,
            reserved_scheduled_tokens=1,
            reserved_kv_token_slots=1,
            regular_aging_steps=10,
        ),
        self_resubmit_policy=SelfResubmitPolicy(
            max_resubmits=1,
            strict_fallback_rolled_back_tokens=100,
            kv_admission_watermark=1.0,
        ),
    )
    scheduler.add("a", token_ids=(1,), max_num_tokens=4, cache_epoch=0)
    scheduler.add("b", token_ids=(2,), max_num_tokens=4, cache_epoch=0)

    for request_id in ("a", "b", "a", "b"):
        output = scheduler.schedule()
        assert output.request_ids == (request_id,)
        scheduler.complete(request_id, num_committed_tokens=1, num_new_tokens=1)

    # a 撞墙并回滚，b 使用空出的页完成；回滚本身不能伪造等待年龄。
    collision = scheduler.schedule()
    assert collision.request_ids == ("b",)
    assert scheduler.stats.self_resubmits_total == 1
    scheduler.complete("b", num_committed_tokens=1, num_new_tokens=1)
    scheduler.remove("b")
    scheduler.add("short", token_ids=(9,), max_num_tokens=2, cache_epoch=0)

    output = scheduler.schedule()

    # 回滚请求即使进入 strict fallback，短请求仍先使用自己的首 token 通道。
    assert output.request_ids[0] == "short"


def test_scheduler_rejects_duplicate_request_ids() -> None:
    scheduler = _scheduler()
    scheduler.add("request", token_ids=(1,), max_num_tokens=5)

    with pytest.raises(SchedulerError, match="already scheduled"):
        scheduler.add("request", token_ids=(1,), max_num_tokens=5)


def test_scheduler_supports_reservations_without_block_placement() -> None:
    scheduler = TokenBudgetScheduler(
        UnboundedKVCacheManager(),
        max_num_sequences=1,
        max_num_scheduled_tokens=2,
    )
    scheduler.add("request", token_ids=(1, 2, 3), max_num_tokens=7)

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
    scheduler.add("request", token_ids=(1,), max_num_tokens=4)

    first = scheduler.schedule().requests[0]
    assert first.num_scheduled_tokens == 1
    assert first.num_lookahead_tokens == 1
    assert first.max_output_tokens == 2

    # 一个输入和一个已接受输出已经写入 KV，bonus token 留到下一轮计算。
    scheduler.complete("request", num_committed_tokens=2, num_new_tokens=2)
    second = scheduler.schedule().requests[0]
    assert second.num_computed_tokens == 2
    assert second.num_scheduled_tokens == 1


def test_scheduler_trims_speculative_budget_at_the_request_length_limit() -> None:
    scheduler = TokenBudgetScheduler(
        PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=4, block_size=1)),
        max_num_sequences=1,
        max_num_scheduled_tokens=4,
        decoding_budget=DecodingBudget(
            num_lookahead_tokens=3,
            max_output_tokens=4,
        ),
    )
    # 请求总共只允许再生成两个 token，不应为第三、第四个结果预留位置。
    scheduler.add("request", token_ids=(1,), max_num_tokens=3)

    scheduled = scheduler.schedule().requests[0]

    assert scheduled.max_output_tokens == 2
    assert scheduled.num_lookahead_tokens == 1
    assert scheduled.block_ids == (0, 1)


def test_scheduler_releases_an_invalid_prefix_match() -> None:
    class InvalidMatchCache:
        def __init__(self) -> None:
            self.freed: list[str] = []

        prefix_caching_enabled = False
        prefix_cache_revision = 0

        def try_add_request(
            self,
            request_id,
            *,
            token_ids,
            max_num_committed_tokens,
            cache_epoch,
            min_free_token_slots=0,
            admission_min_free_token_slots=0,
            initial_extra_blocks=0,
            guarantee_completion=True,
        ):
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
    scheduler.add("request", token_ids=(1, 2), max_num_tokens=6)

    with pytest.raises(SchedulerError, match="leave at least one token"):
        scheduler.schedule()
    assert cache.freed == ["request"]


def test_scheduler_starts_from_a_cached_readonly_prompt_prefix() -> None:
    cache = PagedKVCacheManager(
        FixedKVBlockCapacity(num_blocks=4, block_size=2),
        enable_prefix_caching=True,
    )
    warm_tokens = (1, 2, 3, 4, 5)
    cache.try_add_request(
        "warm",
        token_ids=warm_tokens,
        max_num_committed_tokens=len(warm_tokens),
        cache_epoch=1,
    )
    cache.reserve("warm", len(warm_tokens))
    cache.commit("warm", len(warm_tokens))
    cache.free("warm")
    scheduler = TokenBudgetScheduler(
        cache,
        max_num_sequences=1,
        max_num_scheduled_tokens=2,
    )
    scheduler.add(
        "hit",
        token_ids=(1, 2, 3, 4, 9),
        max_num_tokens=9,
        cache_epoch=1,
    )

    scheduled = scheduler.schedule().requests[0]

    assert scheduled.num_computed_tokens == 4
    assert scheduled.num_scheduled_tokens == 1
    assert scheduled.num_readonly_prefix_blocks == 2
    assert scheduled.block_ids is not None
    assert scheduled.block_ids[:2] == (0, 1)
