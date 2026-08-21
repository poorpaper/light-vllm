"""进程内 PyTorch 模型执行。"""

from __future__ import annotations

import torch

from light_vllm.modeling.models.interfaces import (
    ForwardBatch,
    ModelNotLoadedError,
    ModelSession,
    ModelSessionProvider,
)
from light_vllm.runtime.execution.dense_attention import (
    DenseAttentionMetadata,
    TorchDenseAttention,
)
from light_vllm.runtime.execution.interfaces import (
    ExecutionBatch,
    ExecutionCapabilities,
    ExecutionError,
    ExecutionLease,
    ExecutionNotReadyError,
    ExecutionOutput,
    ExecutionTimer,
    ModelWorker,
    TokenExecutionSession,
)
from light_vllm.runtime.execution.layout import linear_query_layout
from light_vllm.runtime.execution.worker import _forward
from light_vllm.runtime.sampling import Sampler


class _LocalTokenExecutionSession:
    """参考实现为一次生成选定模型后，用它逐个计算新 token。"""

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
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self._device)
        positions = torch.arange(
            len(token_ids),
            dtype=torch.long,
            device=self._device,
        )
        attention = None
        model_spec = self._model.kv_cache_spec
        if model_spec is not None:
            # reference 每轮重算完整序列，但仍使用统一 attention 调用入口。
            attention = TorchDenseAttention(
                model_spec,
                DenseAttentionMetadata(
                    positions=positions,
                    query_layouts=(linear_query_layout(len(token_ids)),),
                ),
            )
        batch = ForwardBatch(
            input_ids=input_ids,
            positions=positions,
            attention=attention,
        )
        output = _forward(self._model, batch)
        if attention is not None:
            expected_layers = frozenset(layer.layer_id for layer in model_spec.layers)
            if attention.layer_ids != expected_layers:
                raise ExecutionError("model did not execute every configured dense attention layer")
        return self._sampler.sample(output.logits[-1:])[0]


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
    """在当前进程中接收 Engine 的调用，并把模型计算交给 Worker。

    以后增加多进程或分布式执行时，可以替换这个 Executor；调度策略不放在这里。
    """

    def __init__(
        self,
        worker: ModelWorker,
        *,
        timer: ExecutionTimer | None = None,
    ) -> None:
        self._worker = worker
        self._timer = timer

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
        if self._timer is None:
            return self._worker.execute(batch)
        output, elapsed_seconds = self._timer.measure(lambda: self._worker.execute(batch))
        return ExecutionOutput(
            requests=output.requests,
            num_model_tokens_computed=output.num_model_tokens_computed,
            step_elapsed_seconds=elapsed_seconds,
        )
