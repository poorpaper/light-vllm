"""进程内 PyTorch 模型执行。"""

from __future__ import annotations

import torch

from light_vllm.modeling.models.interfaces import (
    ForwardBatch,
    ModelNotLoadedError,
    ModelSession,
    ModelSessionProvider,
)
from light_vllm.runtime.execution.interfaces import (
    ExecutionBatch,
    ExecutionCapabilities,
    ExecutionError,
    ExecutionLease,
    ExecutionNotReadyError,
    ExecutionOutput,
    ModelWorker,
    TokenExecutionSession,
)
from light_vllm.runtime.execution.worker import _forward
from light_vllm.runtime.sampling import Sampler


class _LocalTokenExecutionSession:
    """reference 请求在固定模型上的本地执行会话。"""

    def __init__(
        self,
        model: ModelSession,
        sampler: Sampler,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self._model = model
        self._sampler = sampler
        self._device = torch.device(device)

    def next_token(self, token_ids: tuple[int, ...]) -> int:
        if not token_ids:
            raise ExecutionError("token_ids must not be empty")
        batch = ForwardBatch(
            input_ids=torch.tensor([token_ids], dtype=torch.long, device=self._device)
        )
        output = _forward(self._model, batch)
        return self._sampler.sample(output.logits[:, -1])[0]


class LocalTokenExecutor:
    """reference 路径的本地执行器，采样策略通过组合传入。"""

    def __init__(
        self,
        runner: ModelSessionProvider,
        sampler: Sampler,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self._runner = runner
        self._sampler = sampler
        self._device = torch.device(device)

    @property
    def ready(self) -> bool:
        return self._runner.generation > 0

    def open_session(self) -> TokenExecutionSession:
        try:
            model = self._runner.open_session()
        except ModelNotLoadedError as exc:
            raise ExecutionNotReadyError("load a model before executing") from exc
        return _LocalTokenExecutionSession(model, self._sampler, device=self._device)


class LocalModelExecutor:
    """把 Engine 接到单进程执行拓扑，并把设备计算委托给 Worker。

    该层保留未来多进程或分布式 Executor 的扩展位置，不承载调度策略。
    """

    def __init__(self, worker: ModelWorker) -> None:
        self._worker = worker

    @property
    def ready(self) -> bool:
        return self._worker.ready

    @property
    def capabilities(self) -> ExecutionCapabilities:
        return self._worker.capabilities

    def initialize(self) -> None:
        self._worker.initialize()

    def add_request(self, request_id: str, *, capacity: int) -> None:
        self._worker.add_request(request_id, capacity=capacity)

    def free_request(self, request_id: str) -> bool:
        return self._worker.free_request(request_id)

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease:
        return self._worker.acquire(request_ids)

    def execute(self, batch: ExecutionBatch) -> ExecutionOutput:
        return self._worker.execute(batch)
