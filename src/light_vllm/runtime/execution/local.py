"""进程内 PyTorch 模型执行。"""

from __future__ import annotations

import torch

from light_vllm.modeling.models.interfaces import (
    ForwardBatch,
    ModelForwarder,
    ModelNotLoadedError,
    ModelOutput,
)
from light_vllm.runtime.execution.interfaces import (
    ExecutionBatch,
    ExecutionError,
    ExecutionLease,
    ExecutionNotReadyError,
    ExecutionOutput,
    RequestOutput,
)
from light_vllm.runtime.kv_cache import ContiguousKVCache, KVCacheError
from light_vllm.runtime.sampling import Sampler


def _forward(runner: ModelForwarder, batch: ForwardBatch) -> ModelOutput:
    """执行模型边界，并把生命周期错误转换成执行层错误。"""

    try:
        output = runner.forward(batch)
    except ModelNotLoadedError as exc:
        raise ExecutionNotReadyError("load a model before executing") from exc
    if not isinstance(output, ModelOutput):
        raise ExecutionError("model forwarder must return ModelOutput")
    if output.logits.ndim != 3 or output.logits.shape[:2] != batch.input_ids.shape:
        raise ExecutionError("model logits must have shape [batch, sequence, vocabulary]")
    return output


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
    """使用连续 K/V tensor 的统一本地模型执行器。

    它只消费已经可执行的批次，不决定 token budget 或逻辑 block 分配。
    在 Paged Attention 落地前按请求逐个 forward，保持缓存状态转换清楚。
    """

    def __init__(
        self,
        runner: ModelForwarder,
        kv_cache: ContiguousKVCache,
        sampler: Sampler,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self._runner = runner
        self._kv_cache = kv_cache
        self._sampler = sampler
        self._device = torch.device(device)

    @property
    def ready(self) -> bool:
        return self._runner.generation > 0

    def add_request(self, request_id: str, *, capacity: int) -> None:
        self._kv_cache.allocate(request_id, capacity)

    def free_request(self, request_id: str) -> bool:
        return self._kv_cache.free(request_id)

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease:
        return self._kv_cache.acquire(request_ids)

    def execute(self, batch: ExecutionBatch) -> ExecutionOutput:
        results: list[RequestOutput] = []
        for request in batch.requests:
            try:
                cached_tokens = self._kv_cache.cached_tokens(request.request_id)
                if cached_tokens != request.num_computed_tokens:
                    raise ExecutionError(
                        "physical KV length must match the scheduled computed-token count"
                    )
                forward_batch = ForwardBatch(
                    input_ids=torch.tensor(
                        [request.input_token_ids],
                        dtype=torch.long,
                        device=self._device,
                    ),
                    kv_cache=self._kv_cache.view(request.request_id),
                )
                output = _forward(self._runner, forward_batch)
                updates = output.kv_cache_updates
                if updates is None:
                    raise ExecutionError("model did not return KV cache updates")
                if updates.num_tokens != len(request.input_token_ids):
                    raise ExecutionError("model returned the wrong number of KV cache updates")
                self._kv_cache.append(request.request_id, updates)
            except KVCacheError as exc:
                raise ExecutionError(str(exc)) from exc

            token_ids = ()
            if request.sampling_required:
                token_ids = (self._sampler.sample(output.logits[:, -1])[0],)
            results.append(
                RequestOutput(
                    request_id=request.request_id,
                    num_computed_tokens=len(request.input_token_ids),
                    token_ids=token_ids,
                )
            )
        return ExecutionOutput(requests=tuple(results))
