"""进程内模型 Worker 实现。"""

from __future__ import annotations

from threading import RLock

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
from light_vllm.runtime.execution.paged_attention import (
    PagedAttentionMetadata,
    TorchPagedAttention,
)
from light_vllm.runtime.execution.paged_cache import PagedKVCache, PagedKVCacheConfig
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


class ContiguousModelWorker:
    """使用连续 K/V tensor 的进程内 Worker。"""

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

    def initialize(self) -> None:
        if not self.ready:
            raise ExecutionNotReadyError("load a model before initializing the worker")

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


class _PagedExecutionLease:
    """页池在 Worker 关闭前恒定存在，因此执行租约只需保持幂等。"""

    def release(self) -> None:
        return


class PagedModelWorker:
    """使用全局分页 K/V 和同批次 Paged Attention 的进程内 Worker。"""

    def __init__(
        self,
        runner: ModelForwarder,
        sampler: Sampler,
        cache_config: PagedKVCacheConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self._runner = runner
        self._sampler = sampler
        self._cache_config = cache_config
        self._device = torch.device(device)
        if self._device != cache_config.device:
            raise ValueError("worker device must match the paged KV cache device")
        self._cache: PagedKVCache | None = None
        self._model_generation = 0
        self._active_requests: set[str] = set()
        self._lock = RLock()

    @property
    def ready(self) -> bool:
        with self._lock:
            return (
                self._cache is not None
                and self._runner.generation > 0
                and self._model_generation == self._runner.generation
            )

    def initialize(self) -> None:
        model_spec = self._runner.kv_cache_spec
        if self._runner.generation <= 0 or model_spec is None:
            raise ExecutionNotReadyError("load a cacheable model before initializing the worker")
        with self._lock:
            if self._active_requests:
                raise ExecutionError("cannot initialize paged KV cache with active requests")
            if self._model_generation == self._runner.generation and self._cache is not None:
                return
            self._cache = PagedKVCache(model_spec, self._cache_config)
            self._model_generation = self._runner.generation

    def add_request(self, request_id: str, *, capacity: int) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not self.ready:
            raise ExecutionNotReadyError("initialize the worker before adding requests")
        with self._lock:
            if request_id in self._active_requests:
                raise ExecutionError(f"request {request_id!r} already exists in the worker")
            self._active_requests.add(request_id)

    def free_request(self, request_id: str) -> bool:
        with self._lock:
            existed = request_id in self._active_requests
            self._active_requests.discard(request_id)
            return existed

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease:
        request_ids = tuple(request_ids)
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("reserved request IDs must be unique")
        with self._lock:
            missing = tuple(
                request_id for request_id in request_ids if request_id not in self._active_requests
            )
            if missing:
                raise ExecutionError(f"paged worker requests are not active: {missing!r}")
        return _PagedExecutionLease()

    def execute(self, batch: ExecutionBatch) -> ExecutionOutput:
        cache = self._get_cache()
        for request in batch.requests:
            if request.block_ids is None:
                raise ExecutionError("paged worker requires a block table for every request")

        query_width = max(len(request.input_token_ids) for request in batch.requests)
        input_ids = torch.zeros(
            (len(batch.requests), query_width),
            dtype=torch.long,
            device=self._device,
        )
        query_lengths: list[int] = []
        for row, request in enumerate(batch.requests):
            query_length = len(request.input_token_ids)
            query_lengths.append(query_length)
            input_ids[row, :query_length] = torch.tensor(
                request.input_token_ids,
                dtype=torch.long,
                device=self._device,
            )

        metadata = PagedAttentionMetadata(
            block_tables=tuple(request.block_ids or () for request in batch.requests),
            num_computed_tokens=tuple(request.num_computed_tokens for request in batch.requests),
            query_lengths=tuple(query_lengths),
        )
        attention = TorchPagedAttention(cache, metadata)
        output = _forward(
            self._runner,
            ForwardBatch(
                input_ids=input_ids,
                sequence_lengths=tuple(query_lengths),
                attention=attention,
            ),
        )
        expected_layers = frozenset(layer.layer_id for layer in cache.model_spec.layers)
        if attention.layer_ids != expected_layers:
            raise ExecutionError("model did not execute every configured paged attention layer")
        return self._build_output(batch, output, query_lengths)

    def _build_output(
        self,
        batch: ExecutionBatch,
        output: ModelOutput,
        query_lengths: list[int],
    ) -> ExecutionOutput:
        sampling_rows = [
            row for row, request in enumerate(batch.requests) if request.sampling_required
        ]
        sampled: dict[int, int] = {}
        if sampling_rows:
            logits = torch.stack(
                [output.logits[row, query_lengths[row] - 1] for row in sampling_rows]
            )
            sampled_ids = self._sampler.sample(logits)
            sampled = dict(zip(sampling_rows, sampled_ids, strict=True))

        results = tuple(
            RequestOutput(
                request_id=request.request_id,
                num_computed_tokens=len(request.input_token_ids),
                token_ids=(sampled[row],) if row in sampled else (),
            )
            for row, request in enumerate(batch.requests)
        )
        return ExecutionOutput(requests=results)

    def _get_cache(self) -> PagedKVCache:
        with self._lock:
            if not self.ready or self._cache is None:
                raise ExecutionNotReadyError("initialize the worker before executing")
            return self._cache
