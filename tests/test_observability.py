from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from light_vllm import (
    EngineCapabilities,
    GenerateRequest,
    GenerateResult,
    GenerationFinished,
)
from light_vllm.runtime.kv_cache import FixedKVBlockCapacity, PagedKVCacheManager
from light_vllm.runtime.observability import InMemoryPerformanceObserver
from light_vllm.runtime.scheduler import TokenBudgetScheduler
from light_vllm.serving.http import create_http_app


class StubEngineClient:
    ready = True
    capabilities = EngineCapabilities()

    async def stream(self, request: GenerateRequest) -> AsyncIterator[GenerationFinished]:
        yield GenerationFinished(finish_reason="length")

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        return GenerateResult(
            input_ids=request.input_ids,
            generated_token_ids=(),
            finish_reason="length",
        )


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_performance_observer_records_ttft_tpot_tokens_and_outcome() -> None:
    clock = FakeClock()
    observer = InMemoryPerformanceObserver("qwen2.5", clock=clock)

    observer.request_started("request", num_prompt_tokens=3)
    clock.advance(0.2)
    observer.tokens_generated("request", count=1)
    clock.advance(0.1)
    observer.tokens_generated("request", count=2)
    observer.request_finished("request", outcome="finished")

    snapshot = observer.snapshot()
    assert snapshot.time_to_first_token.count == 1
    assert snapshot.time_to_first_token.total == pytest.approx(0.2)
    assert snapshot.time_per_output_token.count == 2
    assert snapshot.time_per_output_token.total == pytest.approx(0.1)
    # 同批第二个 token 的可见间隔是 0；不能把整段时间均摊后扭曲分位数。
    assert snapshot.time_per_output_token.cumulative_counts[0] == 1
    assert snapshot.prompt_tokens_total == 3
    assert snapshot.generation_tokens_total == 3
    assert snapshot.finished_requests_total == 1


def test_scheduler_snapshot_exposes_token_aware_queue_and_kv_usage() -> None:
    cache = PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=4, block_size=2))
    scheduler = TokenBudgetScheduler(
        cache,
        max_num_sequences=1,
        max_num_scheduled_tokens=3,
    )
    scheduler.add("running", token_ids=(1, 2, 3), max_num_tokens=6)
    scheduler.add("waiting", token_ids=(4, 5), max_num_tokens=8)

    queued = scheduler.stats
    assert queued.waiting_requests == 2
    assert queued.waiting_token_budget == 14
    assert queued.running_requests == 0

    scheduler.schedule()
    scheduled = scheduler.stats
    assert scheduled.waiting_requests == 1
    assert scheduled.waiting_token_budget == 8
    assert scheduled.running_requests == 1
    assert scheduled.running_token_budget == 6
    assert scheduled.kv_cache.used_token_slots == 4
    assert scheduled.kv_cache.capacity_token_slots == 8
    assert scheduled.kv_cache.usage_ratio == pytest.approx(0.5)


def test_prefix_cache_counts_active_shared_pages_but_not_evictable_pages() -> None:
    cache = PagedKVCacheManager(
        FixedKVBlockCapacity(num_blocks=4, block_size=2),
        enable_prefix_caching=True,
    )
    cache.add_request("warm", token_ids=(1, 2, 3, 4, 5), cache_epoch=1)
    cache.reserve("warm", 5)
    cache.commit("warm", 5)
    cache.free("warm")

    # 零引用 prefix page 可以立即淘汰，因此不占可用容量。
    assert cache.stats.used_token_slots == 0

    match = cache.add_request(
        "active",
        token_ids=(1, 2, 3, 4, 9),
        cache_epoch=1,
    )
    assert match.num_cached_tokens == 4
    # 被活动请求引用的两个共享页此时不能回收。
    assert cache.stats.used_token_slots == 4


def test_http_metrics_are_ready_for_prometheus_and_hpa() -> None:
    cache = PagedKVCacheManager(FixedKVBlockCapacity(num_blocks=4, block_size=2))
    scheduler = TokenBudgetScheduler(
        cache,
        max_num_sequences=1,
        max_num_scheduled_tokens=2,
    )
    scheduler.add("waiting", token_ids=(1, 2), max_num_tokens=6)
    observer = InMemoryPerformanceObserver("qwen2")
    observer.scheduler_updated(scheduler.stats)
    client = TestClient(create_http_app(StubEngineClient(), performance_metrics=observer))

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert 'light_vllm_requests_waiting{model="qwen2"} 1' in response.text
    assert 'light_vllm_queue_tokens{model="qwen2"} 6' in response.text
    assert "light_vllm_time_to_first_token_seconds_bucket" in response.text
    assert 'light_vllm_requests_total{model="qwen2",outcome="finished"} 0' in response.text
