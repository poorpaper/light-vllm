from __future__ import annotations

import atexit
import json
import os
import signal
import statistics
import sys
import threading
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any

from light_vllm.entrypoints import http as http_entrypoint
from light_vllm.modeling.models.interfaces import ForwardBatch
from light_vllm.runtime.engine import core as engine_core_module
from light_vllm.runtime.engine.core import EngineCore
from light_vllm.runtime.execution import worker as worker_module
from light_vllm.runtime.execution.interfaces import ExecutionBatch, ExecutionRequest
from light_vllm.runtime.execution.local import LocalModelExecutor
from light_vllm.runtime.execution.paged_attention import PagedAttentionMetadata
from light_vllm.runtime.execution.paged_cache import PagedKVCache
from light_vllm.runtime.execution.triton_paged_attention import TritonPagedAttention
from light_vllm.runtime.execution.worker import (
    LocalModelWorker,
    PagedStepHandler,
    StandardDecodeHandler,
    _PagedExecutionLease,
)
from light_vllm.runtime.observability.performance import InMemoryPerformanceObserver
from light_vllm.runtime.sampling import GreedySampler
from light_vllm.runtime.scheduler.interfaces import SchedulerStats
from light_vllm.runtime.scheduler.token_budget import TokenBudgetScheduler
from light_vllm.serving import http as serving_http_module


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _overlap_ns(
    started_ns: int,
    finished_ns: int,
    executor_starts: list[int],
    executor_ends: list[int],
    executor_duration_prefix: list[int],
) -> int:
    first = bisect_right(executor_ends, started_ns)
    last = bisect_left(executor_starts, finished_ns)
    if first >= last:
        return 0
    if first + 1 == last:
        return max(
            0,
            min(finished_ns, executor_ends[first]) - max(started_ns, executor_starts[first]),
        )
    overlap = executor_ends[first] - max(started_ns, executor_starts[first])
    overlap += min(finished_ns, executor_ends[last - 1]) - executor_starts[last - 1]
    overlap += executor_duration_prefix[last - 1] - executor_duration_prefix[first + 1]
    return overlap


class _StageProfiler:
    def __init__(self, output: Path, *, detail: str) -> None:
        self._output = output
        self._detail = detail
        # Signal handlers can interrupt a profiler record call on the main thread.
        # Keep reset reentrant, while the heavier dump runs on a helper thread.
        self._lock = threading.RLock()
        self._active = False
        self._samples: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._executor_intervals: list[tuple[int, int]] = []
        self._executor_records: list[tuple[int, int, int, int | None]] = []
        self._named_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._batch_shapes: list[tuple[int, int, int]] = []
        self._scheduler_shapes: list[tuple[int, int, int, int]] = []
        self._request_lifecycle_ns: dict[str, dict[str, int]] = defaultdict(dict)
        self._cuda_event_ns = 0
        self._reset_wall_ns = 0
        self._reset_process_cpu_ns = 0

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._executor_intervals.clear()
            self._executor_records.clear()
            self._named_intervals.clear()
            self._batch_shapes.clear()
            self._scheduler_shapes.clear()
            self._request_lifecycle_ns.clear()
            self._cuda_event_ns = 0
            self._reset_wall_ns = time.perf_counter_ns()
            self._reset_process_cpu_ns = time.process_time_ns()
            self._active = True

    def record_batch_shape(self, *, batch_size: int, query_width: int, model_tokens: int) -> None:
        if not self._active:
            return
        with self._lock:
            if self._active:
                self._batch_shapes.append((batch_size, query_width, model_tokens))

    def record_scheduler_shape(
        self,
        before: SchedulerStats,
        after: SchedulerStats,
    ) -> None:
        if not self._active:
            return
        with self._lock:
            if self._active:
                self._scheduler_shapes.append(
                    (
                        before.waiting_requests,
                        before.running_requests,
                        after.waiting_requests,
                        after.running_requests,
                    )
                )

    def record_request_event(self, request_id: str, event: str) -> None:
        if not self._active:
            return
        with self._lock:
            if self._active:
                self._request_lifecycle_ns[request_id].setdefault(event, time.perf_counter_ns())

    def record(
        self,
        name: str,
        wall_ns: int,
        cpu_ns: int,
        *,
        started_ns: int | None = None,
    ) -> None:
        if not self._active:
            return
        with self._lock:
            if self._active:
                self._samples[name].append((wall_ns, cpu_ns))
                if started_ns is not None:
                    self._named_intervals[name].append((started_ns, started_ns + wall_ns))

    def record_executor(
        self,
        *,
        started_ns: int,
        finished_ns: int,
        cpu_ns: int,
        cuda_event_seconds: float | None,
    ) -> None:
        if not self._active:
            return
        with self._lock:
            if not self._active:
                return
            self._samples["executor.execute"].append((finished_ns - started_ns, cpu_ns))
            self._executor_intervals.append((started_ns, finished_ns))
            cuda_event_ns = (
                round(cuda_event_seconds * 1_000_000_000)
                if cuda_event_seconds is not None
                else None
            )
            self._executor_records.append((started_ns, finished_ns, cpu_ns, cuda_event_ns))
            if cuda_event_seconds is not None:
                self._cuda_event_ns += cuda_event_ns or 0

    def dump(self) -> None:
        with self._lock:
            if not self._samples:
                return
            self._active = False
            finished_wall_ns = time.perf_counter_ns()
            finished_process_cpu_ns = time.process_time_ns()
            samples = {name: list(values) for name, values in self._samples.items()}
            intervals = list(self._executor_intervals)
            executor_records = list(self._executor_records)
            named_intervals = {name: list(values) for name, values in self._named_intervals.items()}
            batch_shapes = list(self._batch_shapes)
            scheduler_shapes = list(self._scheduler_shapes)
            request_lifecycle_ns = {
                request_id: dict(events)
                for request_id, events in self._request_lifecycle_ns.items()
            }
            cuda_event_ns = self._cuda_event_ns
            reset_wall_ns = self._reset_wall_ns
            reset_process_cpu_ns = self._reset_process_cpu_ns

        stages: dict[str, dict[str, float | int | None]] = {}
        for name, values in sorted(samples.items()):
            wall_ms = [wall_ns / 1_000_000 for wall_ns, _ in values]
            cpu_ms = [cpu_ns / 1_000_000 for _, cpu_ns in values]
            quartile_means = []
            for index in range(4):
                chunk = wall_ms[index * len(wall_ms) // 4 : (index + 1) * len(wall_ms) // 4]
                quartile_means.append(statistics.fmean(chunk) if chunk else None)
            stages[name] = {
                "count": len(values),
                "wall_total_ms": sum(wall_ms),
                "wall_mean_ms": statistics.fmean(wall_ms),
                "wall_p50_ms": _percentile(wall_ms, 0.5),
                "wall_p95_ms": _percentile(wall_ms, 0.95),
                "wall_max_ms": max(wall_ms),
                "thread_cpu_total_ms": sum(cpu_ms),
                "thread_cpu_mean_ms": statistics.fmean(cpu_ms),
                "wall_quartile_means_ms": quartile_means,
            }

        executor_span_ns = 0
        if intervals:
            executor_span_ns = max(end for _, end in intervals) - min(
                start for start, _ in intervals
            )
        executor_wall_ns = round(
            stages.get("executor.execute", {}).get("wall_total_ms", 0.0) * 1_000_000
        )
        sorted_intervals = sorted(intervals)
        executor_starts = [started_ns for started_ns, _ in sorted_intervals]
        executor_ends = [finished_ns for _, finished_ns in sorted_intervals]
        executor_duration_prefix = [0]
        for started_ns, finished_ns in sorted_intervals:
            executor_duration_prefix.append(executor_duration_prefix[-1] + finished_ns - started_ns)
        interval_overlap = {}
        for name, values in named_intervals.items():
            inside_ns = sum(
                _overlap_ns(
                    started_ns,
                    finished_ns,
                    executor_starts,
                    executor_ends,
                    executor_duration_prefix,
                )
                for started_ns, finished_ns in values
            )
            total_ns = sum(finished - started for started, finished in values)
            interval_overlap[name] = {
                "total_ms": total_ns / 1_000_000,
                "inside_executor_ms": inside_ns / 1_000_000,
                "outside_executor_ms": (total_ns - inside_ns) / 1_000_000,
            }
        step_records = []
        previous_finished_ns: int | None = None
        for index, (started_ns, finished_ns, cpu_ns, cuda_ns) in enumerate(executor_records):
            batch_size = query_width = model_tokens = None
            if index < len(batch_shapes):
                batch_size, query_width, model_tokens = batch_shapes[index]
            scheduler_shape = (
                scheduler_shapes[index] if index < len(scheduler_shapes) else (None,) * 4
            )
            step_records.append(
                {
                    "step_id": index,
                    "start_ms": (started_ns - reset_wall_ns) / 1_000_000,
                    "executor_wall_ms": (finished_ns - started_ns) / 1_000_000,
                    "executor_thread_cpu_ms": cpu_ns / 1_000_000,
                    "cuda_event_ms": cuda_ns / 1_000_000 if cuda_ns is not None else None,
                    "previous_executor_gap_ms": (
                        (started_ns - previous_finished_ns) / 1_000_000
                        if previous_finished_ns is not None
                        else None
                    ),
                    "batch_size": batch_size,
                    "query_width": query_width,
                    "model_tokens": model_tokens,
                    "waiting_before": scheduler_shape[0],
                    "running_before": scheduler_shape[1],
                    "waiting_after": scheduler_shape[2],
                    "running_after": scheduler_shape[3],
                }
            )
            previous_finished_ns = finished_ns
        raw_stages = {}
        if self._detail == "full":
            for name, values in samples.items():
                stage_intervals = named_intervals.get(name, ())
                raw_stages[name] = [
                    {
                        "start_ms": (
                            (stage_intervals[index][0] - reset_wall_ns) / 1_000_000
                            if index < len(stage_intervals)
                            else None
                        ),
                        "wall_ms": wall_ns / 1_000_000,
                        "thread_cpu_ms": cpu_ns / 1_000_000,
                    }
                    for index, (wall_ns, cpu_ns) in enumerate(values)
                ]
        payload = {
            "argv": sys.argv,
            "detail": self._detail,
            "profile_window_wall_ms": (finished_wall_ns - reset_wall_ns) / 1_000_000,
            "profile_window_process_cpu_ms": (finished_process_cpu_ns - reset_process_cpu_ns)
            / 1_000_000,
            "executor_steps": len(intervals),
            "executor_span_ms": executor_span_ns / 1_000_000,
            "executor_wall_total_ms": executor_wall_ns / 1_000_000,
            "executor_cuda_event_total_ms": cuda_event_ns / 1_000_000,
            "executor_host_residual_total_ms": (executor_wall_ns - cuda_event_ns) / 1_000_000,
            "inter_executor_gap_total_ms": (executor_span_ns - executor_wall_ns) / 1_000_000,
            "interval_overlap": interval_overlap,
            "batch_shapes": _summarize_batch_shapes(batch_shapes),
            "steps": step_records,
            "request_lifecycle": {
                request_id: {
                    event: (timestamp_ns - reset_wall_ns) / 1_000_000
                    for event, timestamp_ns in events.items()
                }
                for request_id, events in request_lifecycle_ns.items()
            },
            "stages": stages,
            "raw_stages": raw_stages,
        }
        self._output.parent.mkdir(parents=True, exist_ok=True)
        self._output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _summarize_batch_shapes(shapes: list[tuple[int, int, int]]) -> dict[str, object]:
    if not shapes:
        return {}
    padded_tokens = sum(batch_size * query_width for batch_size, query_width, _ in shapes)
    model_tokens = sum(model_tokens for _, _, model_tokens in shapes)
    query_width_histogram: dict[str, int] = defaultdict(int)
    batch_size_histogram: dict[str, int] = defaultdict(int)
    for batch_size, query_width, _ in shapes:
        query_width_histogram[str(query_width)] += 1
        batch_size_histogram[str(batch_size)] += 1
    return {
        "execution_layout": "token-major",
        "steps": len(shapes),
        "model_tokens": model_tokens,
        "executed_query_positions": model_tokens,
        "legacy_padded_positions": padded_tokens,
        "avoided_padding_positions": padded_tokens - model_tokens,
        "legacy_padding_factor": padded_tokens / model_tokens,
        "legacy_padding_waste_percent": 100.0 * (padded_tokens - model_tokens) / padded_tokens,
        # 保留旧字段，便于同一分析脚本读取改造前后的 profile。
        "padded_tokens": padded_tokens,
        "padding_factor": padded_tokens / model_tokens,
        "padding_waste_percent": 100.0 * (padded_tokens - model_tokens) / padded_tokens,
        "decode_width_one_steps": sum(query_width == 1 for _, query_width, _ in shapes),
        "mixed_length_steps": sum(
            model_tokens != batch_size * query_width
            for batch_size, query_width, model_tokens in shapes
        ),
        "query_width_histogram": dict(
            sorted(query_width_histogram.items(), key=lambda item: int(item[0]))
        ),
        "batch_size_histogram": dict(
            sorted(batch_size_histogram.items(), key=lambda item: int(item[0]))
        ),
    }


def _timed_sync(
    profiler: _StageProfiler,
    owner: Any,
    attribute: str,
    stage: str,
) -> None:
    original = getattr(owner, attribute)

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not profiler.active:
            return original(*args, **kwargs)
        wall_started = time.perf_counter_ns()
        cpu_started = time.thread_time_ns()
        try:
            return original(*args, **kwargs)
        finally:
            wall_finished = time.perf_counter_ns()
            profiler.record(
                stage,
                wall_finished - wall_started,
                time.thread_time_ns() - cpu_started,
                started_ns=wall_started,
            )

    setattr(owner, attribute, wrapped)


def _timed_async(
    profiler: _StageProfiler,
    owner: Any,
    attribute: str,
    stage: str,
) -> None:
    original = getattr(owner, attribute)

    @wraps(original)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not profiler.active:
            return await original(*args, **kwargs)
        wall_started = time.perf_counter_ns()
        cpu_started = time.thread_time_ns()
        try:
            return await original(*args, **kwargs)
        finally:
            wall_finished = time.perf_counter_ns()
            profiler.record(
                stage,
                wall_finished - wall_started,
                time.thread_time_ns() - cpu_started,
                started_ns=wall_started,
            )

    setattr(owner, attribute, wrapped)


def _install(profiler: _StageProfiler, *, detail: str) -> None:
    stages = [
        (EngineCore, "_build_execution_batch_locked", "engine.build_batch"),
        (EngineCore, "_apply_output_locked", "engine.apply_output"),
        (EngineCore, "_finish_execution_locked", "engine.finish_execution"),
        (EngineCore, "_publish_scheduler_stats", "engine.publish_scheduler_stats"),
        (LocalModelExecutor, "acquire", "executor.acquire"),
        (_PagedExecutionLease, "release", "executor.lease_release"),
        (InMemoryPerformanceObserver, "step_completed", "observer.step_completed"),
    ]
    if detail == "full":
        stages.extend(
            (
                (TokenBudgetScheduler, "complete", "scheduler.complete"),
                (
                    TokenBudgetScheduler,
                    "_classify_waiting_requests",
                    "scheduler.classify_waiting",
                ),
                (
                    TokenBudgetScheduler,
                    "_admit_waiting_requests",
                    "scheduler.admit_waiting",
                ),
                (
                    TokenBudgetScheduler,
                    "_schedule_admitted_requests",
                    "scheduler.schedule_admitted",
                ),
                (
                    TokenBudgetScheduler,
                    "_schedule_request",
                    "scheduler.schedule_request",
                ),
                (LocalModelWorker, "execute", "worker.execute"),
                (StandardDecodeHandler, "execute", "decode_handler.execute"),
                (PagedStepHandler, "forward", "paged_step.forward"),
                (GreedySampler, "sample", "sampler.sample"),
                (ExecutionRequest, "__post_init__", "execution_request.validate"),
                (ExecutionBatch, "__post_init__", "execution_batch.validate"),
                (ForwardBatch, "__post_init__", "forward_batch.validate"),
                (PagedAttentionMetadata, "__post_init__", "paged_metadata.validate"),
                (
                    PagedAttentionMetadata,
                    "validate_block_tables",
                    "paged_metadata.validate_block_tables",
                ),
                (PagedAttentionMetadata, "slot_mapping", "paged_metadata.slot_mapping"),
                (
                    PagedAttentionMetadata,
                    "visibility_tensor",
                    "paged_metadata.visibility_tensor",
                ),
                (TritonPagedAttention, "__init__", "triton_attention.create"),
                (PagedKVCache, "prepare_write", "paged_cache.prepare_write"),
                (
                    InMemoryPerformanceObserver,
                    "scheduler_updated",
                    "observer.scheduler_updated",
                ),
                (
                    InMemoryPerformanceObserver,
                    "tokens_generated",
                    "observer.tokens_generated",
                ),
            )
        )

    original_paged_forward = PagedStepHandler.forward

    @wraps(original_paged_forward)
    def profiled_paged_forward(self: PagedStepHandler, model: Any, batch: Any) -> Any:
        if profiler.active:
            for request in batch.requests:
                profiler.record_request_event(request.request_id, "first_forward_ms")
            query_lengths = [len(request.query_token_ids) for request in batch.requests]
            profiler.record_batch_shape(
                batch_size=len(query_lengths),
                query_width=max(query_lengths),
                model_tokens=sum(query_lengths),
            )
        return original_paged_forward(self, model, batch)

    PagedStepHandler.forward = profiled_paged_forward
    original_schedule = TokenBudgetScheduler.schedule
    original_add = TokenBudgetScheduler.add

    @wraps(original_add)
    def profiled_add(self: TokenBudgetScheduler, request_id: str, *args: Any, **kwargs: Any) -> Any:
        output = original_add(self, request_id, *args, **kwargs)
        profiler.record_request_event(request_id, "registered_ms")
        return output

    TokenBudgetScheduler.add = profiled_add

    @wraps(original_schedule)
    def profiled_schedule(self: TokenBudgetScheduler) -> Any:
        if not profiler.active:
            return original_schedule(self)
        before = self.stats
        wall_started = time.perf_counter_ns()
        cpu_started = time.thread_time_ns()
        try:
            output = original_schedule(self)
        finally:
            wall_finished = time.perf_counter_ns()
            profiler.record(
                "scheduler.schedule",
                wall_finished - wall_started,
                time.thread_time_ns() - cpu_started,
                started_ns=wall_started,
            )
        profiler.record_scheduler_shape(before, self.stats)
        for request in output.requests:
            profiler.record_request_event(request.request_id, "first_scheduled_ms")
        return output

    TokenBudgetScheduler.schedule = profiled_schedule
    for owner, attribute, stage in stages:
        _timed_sync(profiler, owner, attribute, stage)

    original_apply_output = EngineCore._apply_output_locked

    @wraps(original_apply_output)
    def profiled_apply_output(self: EngineCore, scheduled: Any, output: Any) -> Any:
        for request_id, result in output.items():
            if result.output_token_ids:
                profiler.record_request_event(request_id, "first_token_ready_ms")
        return original_apply_output(self, scheduled, output)

    EngineCore._apply_output_locked = profiled_apply_output
    if hasattr(EngineCore, "_execute_batch"):
        _timed_async(profiler, EngineCore, "_execute_batch", "engine.executor_future_total")

    _timed_sync(profiler, engine_core_module, "_validated_output", "engine.validate_output")
    if detail == "full":
        _timed_sync(profiler, worker_module, "_forward", "worker.model_forward")
        _timed_sync(profiler, serving_http_module, "_encode_event", "http.encode_event")

    original_execute = LocalModelExecutor.execute

    @wraps(original_execute)
    def profiled_execute(self: LocalModelExecutor, *args: Any, **kwargs: Any) -> Any:
        if not profiler.active:
            return original_execute(self, *args, **kwargs)
        wall_started = time.perf_counter_ns()
        cpu_started = time.thread_time_ns()
        output = original_execute(self, *args, **kwargs)
        profiler.record_executor(
            started_ns=wall_started,
            finished_ns=time.perf_counter_ns(),
            cpu_ns=time.thread_time_ns() - cpu_started,
            cuda_event_seconds=output.step_elapsed_seconds,
        )
        return output

    LocalModelExecutor.execute = profiled_execute


def main() -> None:
    raw_output = os.environ.get("LIGHT_VLLM_CPU_PROFILE_OUTPUT")
    if not raw_output:
        raise RuntimeError("LIGHT_VLLM_CPU_PROFILE_OUTPUT must name the JSON output path")
    detail = os.environ.get("LIGHT_VLLM_CPU_PROFILE_DETAIL", "lean")
    if detail not in {"lean", "full"}:
        raise RuntimeError("LIGHT_VLLM_CPU_PROFILE_DETAIL must be lean or full")
    profiler = _StageProfiler(Path(raw_output), detail=detail)
    _install(profiler, detail=detail)

    def reset_profile(_signum: int, _frame: Any) -> None:
        profiler.reset()

    def dump_profile(_signum: int, _frame: Any) -> None:
        # JSON aggregation can take seconds for a full trace.  Returning from the
        # signal handler first keeps the asyncio server responsive and avoids doing
        # lock acquisition, allocation, and filesystem I/O in signal context.
        threading.Thread(
            target=profiler.dump,
            name="light-vllm-profile-dump",
            daemon=True,
        ).start()

    signal.signal(signal.SIGUSR1, reset_profile)
    signal.signal(signal.SIGUSR2, dump_profile)
    atexit.register(profiler.dump)
    http_entrypoint.main()


if __name__ == "__main__":
    main()
