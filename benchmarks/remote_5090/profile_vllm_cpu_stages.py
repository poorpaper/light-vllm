from __future__ import annotations

import atexit
import json
import os
import signal
import statistics
import threading
import time
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


class _Profiler:
    def __init__(self, output: Path) -> None:
        self._output = output
        self._lock = threading.Lock()
        self._active = False
        self._samples: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._step_intervals: list[tuple[int, int]] = []

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._step_intervals.clear()
            self._active = True

    def record(self, name: str, wall_ns: int, cpu_ns: int) -> None:
        if not self._active:
            return
        with self._lock:
            if self._active:
                self._samples[name].append((wall_ns, cpu_ns))

    def record_step(self, *, started_ns: int, finished_ns: int, cpu_ns: int) -> None:
        if not self._active:
            return
        with self._lock:
            if not self._active:
                return
            self._samples["engine.step"].append((finished_ns - started_ns, cpu_ns))
            self._step_intervals.append((started_ns, finished_ns))

    def dump(self) -> None:
        with self._lock:
            if not self._samples:
                return
            self._active = False
            samples = {name: list(values) for name, values in self._samples.items()}
            intervals = list(self._step_intervals)

        stages: dict[str, dict[str, float | int | None]] = {}
        for name, values in sorted(samples.items()):
            wall_ms = [wall_ns / 1_000_000 for wall_ns, _ in values]
            cpu_ms = [cpu_ns / 1_000_000 for _, cpu_ns in values]
            stages[name] = {
                "count": len(values),
                "wall_total_ms": sum(wall_ms),
                "wall_mean_ms": statistics.fmean(wall_ms),
                "wall_p50_ms": _percentile(wall_ms, 0.5),
                "wall_p95_ms": _percentile(wall_ms, 0.95),
                "wall_max_ms": max(wall_ms),
                "thread_cpu_total_ms": sum(cpu_ms),
                "thread_cpu_mean_ms": statistics.fmean(cpu_ms),
            }

        step_span_ns = 0
        if intervals:
            step_span_ns = max(end for _, end in intervals) - min(start for start, _ in intervals)
        step_wall_ns = round(stages.get("engine.step", {}).get("wall_total_ms", 0.0) * 1_000_000)
        payload = {
            "pid": os.getpid(),
            "engine_steps": len(intervals),
            "engine_step_span_ms": step_span_ns / 1_000_000,
            "engine_step_wall_total_ms": step_wall_ns / 1_000_000,
            "inter_step_gap_total_ms": (step_span_ns - step_wall_ns) / 1_000_000,
            "stages": stages,
        }
        self._output.parent.mkdir(parents=True, exist_ok=True)
        self._output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _timed_sync(profiler: _Profiler, owner: Any, attribute: str, stage: str) -> None:
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
            profiler.record(
                stage,
                time.perf_counter_ns() - wall_started,
                time.thread_time_ns() - cpu_started,
            )

    setattr(owner, attribute, wrapped)


def _install() -> None:
    raw_output = os.environ.get("VLLM_CPU_PROFILE_OUTPUT")
    if not raw_output:
        return

    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.uniproc_executor import AsyncOutputFuture, UniProcExecutor

    output = Path(raw_output.format(pid=os.getpid()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".pid").write_text(str(os.getpid()), encoding="utf-8")
    profiler = _Profiler(output)
    for owner, attribute, stage in (
        (Scheduler, "schedule", "scheduler.schedule"),
        (Scheduler, "get_grammar_bitmask", "scheduler.grammar_bitmask"),
        (Scheduler, "update_from_output", "scheduler.update_from_output"),
        (UniProcExecutor, "execute_model", "executor.execute_model"),
        (UniProcExecutor, "sample_tokens", "executor.sample_tokens"),
        (AsyncOutputFuture, "result", "executor.future_result"),
        (EngineCore, "_process_aborts_queue", "engine.process_aborts"),
        (EngineCore, "_attach_iteration_details", "engine.attach_iteration_details"),
        (EngineCore, "post_step", "engine.post_step"),
    ):
        _timed_sync(profiler, owner, attribute, stage)

    def wrap_step(attribute: str, stage: str) -> None:
        original = getattr(EngineCore, attribute)

        @wraps(original)
        def wrapped(self: EngineCore, *args: Any, **kwargs: Any) -> Any:
            if not profiler.active:
                return original(self, *args, **kwargs)
            wall_started = time.perf_counter_ns()
            cpu_started = time.thread_time_ns()
            try:
                return original(self, *args, **kwargs)
            finally:
                finished_ns = time.perf_counter_ns()
                profiler.record_step(
                    started_ns=wall_started,
                    finished_ns=finished_ns,
                    cpu_ns=time.thread_time_ns() - cpu_started,
                )
                if stage != "engine.step":
                    profiler.record(
                        stage,
                        finished_ns - wall_started,
                        time.thread_time_ns() - cpu_started,
                    )

        setattr(EngineCore, attribute, wrapped)

    wrap_step("step", "engine.step")
    wrap_step("step_with_batch_queue", "engine.step_with_batch_queue")

    def reset_profile(_signum: int, _frame: Any) -> None:
        profiler.reset()

    def dump_profile(_signum: int, _frame: Any) -> None:
        profiler.dump()

    signal.signal(signal.SIGUSR1, reset_profile)
    signal.signal(signal.SIGUSR2, dump_profile)
    atexit.register(profiler.dump)


_install()
