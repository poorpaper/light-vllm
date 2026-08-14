"""进程内 PyTorch 模型执行。"""

from __future__ import annotations

import torch

from light_vllm.modeling.models.interfaces import ForwardBatch, ModelForwarder
from light_vllm.runtime.execution.interfaces import (
    ExecutionBatch,
    ExecutionError,
    ExecutionLease,
    ExecutionOutput,
    ModelWorker,
)
from light_vllm.runtime.execution.worker import _forward
from light_vllm.runtime.sampling import Sampler


class LocalTokenExecutor:
    """reference 路径的本地执行器，采样策略通过组合传入。"""

    def __init__(
        self,
        runner: ModelForwarder,
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

    def next_token(self, token_ids: tuple[int, ...]) -> int:
        if not token_ids:
            raise ExecutionError("token_ids must not be empty")
        batch = ForwardBatch(
            input_ids=torch.tensor([token_ids], dtype=torch.long, device=self._device)
        )
        output = _forward(self._runner, batch)
        return self._sampler.sample(output.logits[:, -1])[0]


class LocalModelExecutor:
    """把 Engine 的执行端口接到一个进程内 Worker。"""

    def __init__(self, worker: ModelWorker) -> None:
        self._worker = worker

    @property
    def ready(self) -> bool:
        return self._worker.ready

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
