from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from light_vllm.runtime.generation.interfaces import (
    GenerateRequest,
    GenerateResult,
    GenerationEvent,
)
from light_vllm.runtime.observability.interfaces import StepObservation
from light_vllm.runtime.scheduler.interfaces import SchedulerStats


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    """引擎对外报告的模型长度、KV cache 和调度容量上限。"""

    max_model_tokens: int | None = None
    max_kv_cache_tokens: int | None = None
    max_num_sequences: int = 1
    max_num_scheduled_tokens: int | None = None

    @property
    def max_request_tokens(self) -> int | None:
        """一个请求最多能使用的 token 数，不表示此刻还有多少空闲容量。"""

        limits = tuple(
            limit
            for limit in (self.max_model_tokens, self.max_kv_cache_tokens)
            if limit is not None
        )
        return min(limits) if limits else None


class RequestAdmission(Protocol):
    """请求进入调度队列前，检查它是否可能被当前引擎处理。"""

    def validate(self, request: GenerateRequest, capabilities: EngineCapabilities) -> None: ...


class StepLatencyPredictor(Protocol):
    """根据待处理 token 量估计首 token 前的计算延迟。"""

    def predict(self, num_pending_tokens: int) -> float | None: ...

    def observe(self, observation: StepObservation) -> None: ...


class TTFTAdmission(Protocol):
    """依据实时调度负载做可选的 TTFT SLO 准入。"""

    def validate(self, request: GenerateRequest, stats: SchedulerStats) -> None: ...

    def step_completed(self, observation: StepObservation) -> None: ...


class EngineClient(Protocol):
    """HTTP/RPC 调用推理引擎的统一异步接口。"""

    @property
    def ready(self) -> bool: ...

    @property
    def capabilities(self) -> EngineCapabilities: ...

    def stream(self, request: GenerateRequest) -> AsyncIterator[GenerationEvent]: ...

    async def generate(self, request: GenerateRequest) -> GenerateResult: ...
