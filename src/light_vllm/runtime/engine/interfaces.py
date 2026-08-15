from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from light_vllm.runtime.generation.interfaces import (
    GenerateRequest,
    GenerateResult,
    GenerationEvent,
)


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    """模型、KV 和 Scheduler 初始化后对外暴露的静态能力事实。"""

    max_model_tokens: int | None = None
    max_kv_cache_tokens: int | None = None
    max_num_sequences: int = 1
    max_num_scheduled_tokens: int | None = None

    @property
    def max_request_tokens(self) -> int | None:
        """单请求确定性上限，不代表此刻剩余的可用容量。"""

        limits = tuple(
            limit
            for limit in (self.max_model_tokens, self.max_kv_cache_tokens)
            if limit is not None
        )
        return min(limits) if limits else None


class RequestAdmission(Protocol):
    """在请求进入 Scheduler 前检查确定性的容量限制。"""

    def validate(self, request: GenerateRequest, capabilities: EngineCapabilities) -> None: ...


class EngineClient(Protocol):
    """HTTP/RPC 调用推理引擎的统一异步接口。"""

    @property
    def ready(self) -> bool: ...

    @property
    def capabilities(self) -> EngineCapabilities: ...

    def stream(self, request: GenerateRequest) -> AsyncIterator[GenerationEvent]: ...

    async def generate(self, request: GenerateRequest) -> GenerateResult: ...
