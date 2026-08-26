"""按 Rank 记录 TP 服务热路径的 CPU 墙钟时间。

该脚本只用于远端归因，不进入正常 serving 路径。先向各 Rank 发送 SIGUSR1
开始记录，跑完目标负载后发送 SIGUSR2，即可得到模型执行、控制通信和步间
空档的分层数据。
"""

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

from light_vllm.entrypoints import http as http_entrypoint
from light_vllm.runtime.engine import core as engine_core_module
from light_vllm.runtime.engine.core import EngineCore
from light_vllm.runtime.engine.execution_lane import ExecutionLane
from light_vllm.runtime.execution.distributed import (
    TensorParallelModelExecutor,
    TorchDistributedGroup,
    _GlooCommandChannel,
    _SocketCommandChannel,
    _TensorParallelLease,
)
from light_vllm.runtime.execution.local import LocalModelExecutor
from light_vllm.runtime.execution.worker import (
    LocalModelWorker,
    PagedStepHandler,
    StandardDecodeHandler,
)
from light_vllm.runtime.observability.performance import InMemoryPerformanceObserver
from light_vllm.runtime.sampling import ConfigurableSampler
from light_vllm.runtime.scheduler.token_budget import TokenBudgetScheduler


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


class _Profiler:
    def __init__(self, output: Path) -> None:
        self._output = output
        self._lock = threading.RLock()
        self._active = False
        self._samples: dict[str, list[tuple[int, int, int]]] = defaultdict(list)

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._active = True

    def record(self, name: str, started_ns: int, finished_ns: int, cpu_ns: int) -> None:
        if not self._active:
            return
        with self._lock:
            if self._active:
                self._samples[name].append((started_ns, finished_ns, cpu_ns))

    def dump(self) -> None:
        with self._lock:
            if not self._samples:
                return
            self._active = False
            samples = {name: list(values) for name, values in self._samples.items()}

        stages: dict[str, dict[str, float | int]] = {}
        for name, values in sorted(samples.items()):
            wall_ms = [(finished - started) / 1_000_000 for started, finished, _ in values]
            cpu_ms = [cpu / 1_000_000 for _, _, cpu in values]
            stages[name] = {
                "count": len(values),
                "wall_total_ms": sum(wall_ms),
                "wall_mean_ms": statistics.fmean(wall_ms),
                "wall_p50_ms": _percentile(wall_ms, 0.5),
                "wall_p95_ms": _percentile(wall_ms, 0.95),
                "thread_cpu_mean_ms": statistics.fmean(cpu_ms),
            }

        executor_intervals = samples.get("tp_executor.execute", ())
        inter_step_ms: list[float] = []
        for previous, current in zip(executor_intervals, executor_intervals[1:], strict=False):
            inter_step_ms.append((current[0] - previous[1]) / 1_000_000)
        payload: dict[str, Any] = {
            "rank": int(os.environ.get("RANK", "0")),
            "pid": os.getpid(),
            "stages": stages,
        }
        if inter_step_ms:
            payload["inter_step_gap"] = {
                "count": len(inter_step_ms),
                "mean_ms": statistics.fmean(inter_step_ms),
                "p50_ms": _percentile(inter_step_ms, 0.5),
                "p95_ms": _percentile(inter_step_ms, 0.95),
            }
        self._output.parent.mkdir(parents=True, exist_ok=True)
        self._output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _timed_sync(profiler: _Profiler, owner: Any, attribute: str, name: str) -> None:
    original = getattr(owner, attribute)

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not profiler.active:
            return original(*args, **kwargs)
        started_ns = time.perf_counter_ns()
        cpu_started_ns = time.thread_time_ns()
        try:
            return original(*args, **kwargs)
        finally:
            profiler.record(
                name,
                started_ns,
                time.perf_counter_ns(),
                time.thread_time_ns() - cpu_started_ns,
            )

    setattr(owner, attribute, wrapped)


def _timed_async(profiler: _Profiler, owner: Any, attribute: str, name: str) -> None:
    original = getattr(owner, attribute)

    @wraps(original)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not profiler.active:
            return await original(*args, **kwargs)
        started_ns = time.perf_counter_ns()
        cpu_started_ns = time.thread_time_ns()
        try:
            return await original(*args, **kwargs)
        finally:
            profiler.record(
                name,
                started_ns,
                time.perf_counter_ns(),
                time.thread_time_ns() - cpu_started_ns,
            )

    setattr(owner, attribute, wrapped)


def main() -> None:
    raw_output = os.environ.get("LIGHT_VLLM_TP_PROFILE_OUTPUT")
    if not raw_output:
        raise RuntimeError("LIGHT_VLLM_TP_PROFILE_OUTPUT must contain a {rank} placeholder")
    rank = int(os.environ.get("RANK", "0"))
    output = Path(raw_output.format(rank=rank))
    profiler = _Profiler(output)

    for owner, attribute, name in (
        (TensorParallelModelExecutor, "execute", "tp_executor.execute"),
        (TensorParallelModelExecutor, "_run_command", "tp_executor.run_command"),
        (TorchDistributedGroup, "broadcast_object", "control.broadcast_object"),
        (TorchDistributedGroup, "first_rank", "control.status_all_reduce"),
        (_GlooCommandChannel, "broadcast", "control.gloo.broadcast"),
        (_GlooCommandChannel, "complete", "control.gloo.complete"),
        (_SocketCommandChannel, "broadcast", "control.socket.broadcast"),
        (_SocketCommandChannel, "complete", "control.socket.complete"),
        (LocalModelExecutor, "execute", "local_executor.execute"),
        (LocalModelWorker, "execute", "worker.execute"),
        (StandardDecodeHandler, "execute", "decode.execute"),
        (PagedStepHandler, "forward", "model_step.forward"),
        (ConfigurableSampler, "sample", "sampler.sample"),
        (EngineCore, "_advance_locked", "engine.advance_locked"),
        (EngineCore, "_apply_output_locked", "engine.apply_output"),
        (EngineCore, "_finish_execution_locked", "engine.finish_execution"),
        (EngineCore, "_build_execution_batch_locked", "engine.build_batch"),
        (EngineCore, "_publish_scheduler_stats", "engine.publish_scheduler_stats"),
        (TokenBudgetScheduler, "schedule", "scheduler.schedule"),
        (TokenBudgetScheduler, "complete", "scheduler.complete"),
        (TensorParallelModelExecutor, "acquire", "tp_executor.acquire"),
        (_TensorParallelLease, "release", "tp_executor.lease_release"),
        (ExecutionLane, "_finish", "execution_lane.finish"),
        (InMemoryPerformanceObserver, "step_completed", "observer.step_completed"),
        (InMemoryPerformanceObserver, "scheduler_updated", "observer.scheduler_updated"),
        (InMemoryPerformanceObserver, "tokens_generated", "observer.tokens_generated"),
        (engine_core_module, "_validated_output", "engine.validate_output"),
    ):
        _timed_sync(profiler, owner, attribute, name)
    _timed_async(profiler, ExecutionLane, "execute", "execution_lane.execute")
    _timed_async(profiler, EngineCore, "_execute_batch", "engine.execute_batch")
    _timed_async(
        profiler,
        EngineCore,
        "_execute_prepared_step",
        "engine.execute_prepared_step",
    )

    pid_file = output.with_suffix(".pid")
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()), encoding="utf-8")

    def reset_profile(_signum: int, _frame: Any) -> None:
        profiler.reset()

    def dump_profile(_signum: int, _frame: Any) -> None:
        threading.Thread(target=profiler.dump, daemon=True).start()

    signal.signal(signal.SIGUSR1, reset_profile)
    signal.signal(signal.SIGUSR2, dump_profile)
    atexit.register(profiler.dump)
    if os.environ.get("LIGHT_VLLM_TP_PROFILE_AUTO_START") == "1":
        profiler.reset()
    http_entrypoint.main()


if __name__ == "__main__":
    main()
