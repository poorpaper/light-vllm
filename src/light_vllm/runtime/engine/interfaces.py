from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from light_vllm.runtime.generation.interfaces import (
    GenerateRequest,
    GenerateResult,
    GenerationEvent,
)


class EngineClient(Protocol):
    """HTTP/RPC 调用推理引擎的统一异步接口。"""

    @property
    def ready(self) -> bool: ...

    def stream(self, request: GenerateRequest) -> AsyncIterator[GenerationEvent]: ...

    async def generate(self, request: GenerateRequest) -> GenerateResult: ...
