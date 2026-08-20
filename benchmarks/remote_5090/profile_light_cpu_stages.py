from __future__ import annotations

import asyncio
import atexit
import json
import os
import signal
import statistics
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

from light_vllm.entrypoints import http as http_entrypoint
from light_vllm.runtime.engine import core as engine_core_module
from light_vllm.runtime.engine.core import EngineCore
from light_vllm.runtime.execution.interfaces import ExecutionBatch, ExecutionRequest
from light_vllm.runtime.execution.local import LocalModelExecutor
from light_vllm.runtime.execution.worker import (
    LocalModelWorker,
    PagedStepHandler,
    StandardDecodeHandler,
    _PagedExecutionLease,
)
from light_vllm.runtime.observability.performance import InMemoryPerformanceObserver
from light_vllm.runtime.sampling import GreedySampler
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


class _StageProfiler:
    def __init__(self, output: Path, *, detail: str) -> None:
        self._output = output
        self._detail = detail
        self._lock = threading.Lock()
        self._active = False
        self._samples: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._executor_intervals: list[tuple[int, int]] = []
        self._named_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
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
            self._named_intervals.clear()
            self._cuda_event_ns = 0
            self._reset_wall_ns = time.perf_counter_ns()
            self._reset_process_cpu_ns = time.process_time_ns()
            self._active = True

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
            if cuda_event_seconds is not None:
                self._cuda_event_ns += round(cuda_event_seconds * 1_000_000_000)

    def dump(self) -> None:
        with self._lock:
            if not self._samples:
                return
            self._active = False
            finished_wall_ns = time.perf_counter_ns()
            finished_process_cpu_ns = time.process_time_ns()
            samples = {name: list(values) for name, values in self._samples.items()}
            intervals = list(self._executor_intervals)
            named_intervals = {name: list(values) for name, values in self._named_intervals.items()}
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
        interval_overlap = {}
        for name, values in named_intervals.items():
            inside_ns = 0
            for started_ns, finished_ns in values:
                inside_ns += sum(
                    max(0, min(finished_ns, executor_end) - max(started_ns, executor_start))
                    for executor_start, executor_end in intervals
                )
            total_ns = sum(finished - started for started, finished in values)
            interval_overlap[name] = {
                "total_ms": total_ns / 1_000_000,
                "inside_executor_ms": inside_ns / 1_000_000,
                "outside_executor_ms": (total_ns - inside_ns) / 1_000_000,
            }
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
            "stages": stages,
        }
        self._output.parent.mkdir(parents=True, exist_ok=True)
        self._output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def _install(profiler: _StageProfiler, *, detail: str) -> None:
    stages = [
        (TokenBudgetScheduler, "schedule", "scheduler.schedule"),
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
                (LocalModelWorker, "execute", "worker.execute"),
                (StandardDecodeHandler, "execute", "decode_handler.execute"),
                (PagedStepHandler, "forward", "paged_step.forward"),
                (GreedySampler, "sample", "sampler.sample"),
                (ExecutionRequest, "__post_init__", "execution_request.validate"),
                (ExecutionBatch, "__post_init__", "execution_batch.validate"),
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
    for owner, attribute, stage in stages:
        _timed_sync(profiler, owner, attribute, stage)

    _timed_sync(profiler, engine_core_module, "_validated_output", "engine.validate_output")
    if detail == "full":
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

    original_to_thread: Callable[..., Any] = asyncio.to_thread

    @wraps(original_to_thread)
    async def profiled_to_thread(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        is_executor = getattr(func, "__name__", None) == "execute" and isinstance(
            getattr(func, "__self__", None), LocalModelExecutor
        )
        if not profiler.active or not is_executor:
            return await original_to_thread(func, *args, **kwargs)
        wall_started = time.perf_counter_ns()
        cpu_started = time.thread_time_ns()
        try:
            return await original_to_thread(func, *args, **kwargs)
        finally:
            wall_finished = time.perf_counter_ns()
            profiler.record(
                "engine.to_thread_total",
                wall_finished - wall_started,
                time.thread_time_ns() - cpu_started,
                started_ns=wall_started,
            )

    asyncio.to_thread = profiled_to_thread


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
        profiler.dump()

    signal.signal(signal.SIGUSR1, reset_profile)
    signal.signal(signal.SIGUSR2, dump_profile)
    atexit.register(profiler.dump)
    http_entrypoint.main()


if __name__ == "__main__":
    main()
