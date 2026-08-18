"""隔离一个或多个性能观察者，避免旁路故障影响推理。"""

from __future__ import annotations

from threading import Lock

from light_vllm.runtime.observability.interfaces import (
    PerformanceObserver,
    RequestOutcome,
    StepObservation,
)
from light_vllm.runtime.scheduler.interfaces import SchedulerStats


class SafeCompositePerformanceObserver:
    """逐个通知观察者；某个观察者首次失败后立即停用。"""

    def __init__(self, *observers: PerformanceObserver | None) -> None:
        self._lock = Lock()
        self._observers = [observer for observer in observers if observer is not None]

    def request_started(self, request_id: str, *, num_prompt_tokens: int) -> None:
        self._notify("request_started", request_id, num_prompt_tokens=num_prompt_tokens)

    def tokens_generated(self, request_id: str, *, count: int) -> None:
        self._notify("tokens_generated", request_id, count=count)

    def request_finished(self, request_id: str, *, outcome: RequestOutcome) -> None:
        self._notify("request_finished", request_id, outcome=outcome)

    def scheduler_updated(self, stats: SchedulerStats) -> None:
        self._notify("scheduler_updated", stats)

    def step_completed(self, observation: StepObservation) -> None:
        self._notify("step_completed", observation)

    def _notify(self, method_name: str, *args: object, **kwargs: object) -> None:
        with self._lock:
            observers = tuple(self._observers)
        for observer in observers:
            try:
                getattr(observer, method_name)(*args, **kwargs)
            except Exception:
                # 不在 Engine 热路径写日志；失败的旁路实现只停用一次。
                with self._lock:
                    if observer in self._observers:
                        self._observers.remove(observer)
