"""进程内模型 Worker 实现。"""

from __future__ import annotations

from threading import RLock

import torch

from light_vllm.modeling.models.interfaces import (
    ForwardBatch,
    ModelNotLoadedError,
    ModelOutput,
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
    RequestOutput,
)
from light_vllm.runtime.execution.paged_attention import (
    PagedAttentionBackend,
    PagedAttentionMetadata,
)
from light_vllm.runtime.execution.paged_cache import PagedKVCache, PagedKVCachePlanner
from light_vllm.runtime.kv_cache import (
    ContiguousKVCache,
    ContiguousKVCacheConfig,
    KVCacheError,
)
from light_vllm.runtime.sampling import Sampler


def _forward(model: ModelSession, batch: ForwardBatch) -> ModelOutput:
    """执行模型边界，并把生命周期错误转换成执行层错误。"""

    try:
        output = model.forward(batch)
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
        runner: ModelSessionProvider,
        sampler: Sampler,
        cache_config: ContiguousKVCacheConfig,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self._runner = runner
        self._sampler = sampler
        self._cache_config = cache_config
        self._device = torch.device(device)
        if self._device != cache_config.device:
            raise ValueError("worker device must match the contiguous KV cache device")
        self._model: ModelSession | None = None
        self._kv_cache: ContiguousKVCache | None = None
        self._lock = RLock()

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._model is not None and self._model.generation == self._runner.generation

    @property
    def capabilities(self) -> ExecutionCapabilities:
        _, model = self._get_runtime()
        return ExecutionCapabilities(
            max_model_tokens=model.max_model_tokens,
            max_kv_cache_tokens=None,
        )

    def initialize(self) -> None:
        try:
            model = self._runner.open_session()
        except ModelNotLoadedError as exc:
            raise ExecutionNotReadyError("load a model before initializing the worker") from exc
        model_spec = model.kv_cache_spec
        if model_spec is None:
            raise ExecutionNotReadyError("load a cacheable model before initializing the worker")
        # 连续缓存的层数和 head 形状只从固定 session 的模型规格获得。
        candidate = ContiguousKVCache(model_spec, self._cache_config)

        with self._lock:
            if self._kv_cache is not None and self._kv_cache.num_requests:
                raise ExecutionError("cannot initialize contiguous KV cache with active requests")
            if self._runner.generation != model.generation:
                raise ExecutionError("model changed while initializing the contiguous KV cache")
            if self._model is not None and self._model.generation == model.generation:
                return
            self._model = model
            self._kv_cache = candidate

    def add_request(self, request_id: str, *, capacity: int) -> None:
        if not self.ready:
            raise ExecutionNotReadyError("initialize the worker before adding requests")
        cache, _ = self._get_runtime()
        cache.allocate(request_id, capacity)

    def free_request(self, request_id: str) -> bool:
        cache, _ = self._get_runtime()
        return cache.free(request_id)

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease:
        cache, _ = self._get_runtime()
        return cache.acquire(request_ids)

    def execute(self, batch: ExecutionBatch) -> ExecutionOutput:
        cache, model = self._get_runtime()
        results: list[RequestOutput] = []
        for request in batch.requests:
            # 当前 correctness Worker 只实现普通输出；投机 Worker 将消费 lookahead。
            if request.num_lookahead_tokens:
                raise ExecutionError("contiguous worker does not consume lookahead tokens")
            try:
                cached_tokens = cache.cached_tokens(request.request_id)
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
                    positions=torch.arange(
                        cached_tokens,
                        cached_tokens + len(request.input_token_ids),
                        dtype=torch.long,
                        device=self._device,
                    ).unsqueeze(0),
                    kv_cache=cache.view(request.request_id),
                )
                output = _forward(model, forward_batch)
                updates = output.kv_cache_updates
                if updates is None:
                    raise ExecutionError("model did not return KV cache updates")
                if updates.num_tokens != len(request.input_token_ids):
                    raise ExecutionError("model returned the wrong number of KV cache updates")
                cache.append(request.request_id, updates)
            except KVCacheError as exc:
                raise ExecutionError(str(exc)) from exc

            token_ids = ()
            if request.max_output_tokens:
                token_ids = (self._sampler.sample(output.logits[:, -1])[0],)
            results.append(
                RequestOutput(
                    request_id=request.request_id,
                    num_input_tokens_computed=len(request.input_token_ids),
                    output_token_ids=token_ids,
                )
            )
        return ExecutionOutput(requests=tuple(results))

    def _get_runtime(self) -> tuple[ContiguousKVCache, ModelSession]:
        with self._lock:
            if self._kv_cache is None or self._model is None:
                raise ExecutionNotReadyError("initialize the worker before executing")
            return self._kv_cache, self._model


class _PagedExecutionLease:
    """页池在 Worker 关闭前恒定存在，因此执行租约只需保持幂等。"""

    def release(self) -> None:
        return


class PagedModelWorker:
    """使用全局分页 K/V 和同批次 Paged Attention 的进程内 Worker。"""

    def __init__(
        self,
        runner: ModelSessionProvider,
        sampler: Sampler,
        cache_planner: PagedKVCachePlanner,
        attention_backend: PagedAttentionBackend,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self._runner = runner
        self._sampler = sampler
        self._cache_planner = cache_planner
        self._attention_backend = attention_backend
        self._device = torch.device(device)
        if self._device != cache_planner.device:
            raise ValueError("worker device must match the paged KV cache device")
        self._cache: PagedKVCache | None = None
        self._model: ModelSession | None = None
        self._active_requests: set[str] = set()
        self._lock = RLock()

    @property
    def ready(self) -> bool:
        with self._lock:
            return (
                self._cache is not None
                and self._model is not None
                and self._model.generation == self._runner.generation
            )

    @property
    def capabilities(self) -> ExecutionCapabilities:
        cache, model = self._get_runtime()
        return ExecutionCapabilities(
            max_model_tokens=model.max_model_tokens,
            max_kv_cache_tokens=(cache.config.num_blocks * cache.config.block_size),
        )

    def initialize(self) -> None:
        with self._lock:
            # 活动请求仍引用旧 generation 的页池，不能原地替换。
            if self._active_requests:
                raise ExecutionError("cannot initialize paged KV cache with active requests")
        try:
            model = self._runner.open_session()
        except ModelNotLoadedError as exc:
            raise ExecutionNotReadyError(
                "load a cacheable model before initializing the worker"
            ) from exc
        model_spec = model.kv_cache_spec
        if model_spec is None:
            raise ExecutionNotReadyError("load a cacheable model before initializing the worker")
        # 模型加载后才能用真实 KV 规格和剩余显存解析物理页容量。
        cache_config = self._cache_planner.plan(model_spec)
        candidate = PagedKVCache(model_spec, cache_config)

        with self._lock:
            # 页池在锁外构造；安装前再次核对请求状态和模型 generation。
            if self._active_requests:
                raise ExecutionError("cannot initialize paged KV cache with active requests")
            if self._runner.generation != model.generation:
                raise ExecutionError("model changed while initializing the paged KV cache")
            if self._model is not None and self._model.generation == model.generation:
                return
            self._cache = candidate
            self._model = model

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
        cache, model = self._get_runtime()
        for request in batch.requests:
            if request.block_ids is None:
                raise ExecutionError("paged worker requires a block table for every request")
            if request.num_lookahead_tokens:
                raise ExecutionError("paged worker does not consume lookahead tokens")

        query_width = max(len(request.input_token_ids) for request in batch.requests)
        input_ids = torch.zeros(
            (len(batch.requests), query_width),
            dtype=torch.long,
            device=self._device,
        )
        positions = torch.zeros_like(input_ids)
        query_lengths: list[int] = []
        for row, request in enumerate(batch.requests):
            query_length = len(request.input_token_ids)
            query_lengths.append(query_length)
            input_ids[row, :query_length] = torch.tensor(
                request.input_token_ids,
                dtype=torch.long,
                device=self._device,
            )
            positions[row, :query_length] = torch.arange(
                request.num_computed_tokens,
                request.num_computed_tokens + query_length,
                dtype=torch.long,
                device=self._device,
            )

        metadata = PagedAttentionMetadata(
            block_tables=tuple(request.block_ids or () for request in batch.requests),
            num_computed_tokens=tuple(request.num_computed_tokens for request in batch.requests),
            query_lengths=tuple(query_lengths),
        )
        # Worker 依赖 backend factory；模型最终只看到通用 AttentionContext。
        attention = self._attention_backend.create(cache, metadata)
        output = _forward(
            model,
            ForwardBatch(
                input_ids=input_ids,
                positions=positions,
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
            row for row, request in enumerate(batch.requests) if request.max_output_tokens
        ]
        sampled: dict[int, int] = {}
        if sampling_rows:
            # 每行只从最后一个有效 query 位置采样，不能读取右侧 padding。
            logits = torch.stack(
                [output.logits[row, query_lengths[row] - 1] for row in sampling_rows]
            )
            sampled_ids = self._sampler.sample(logits)
            sampled = dict(zip(sampling_rows, sampled_ids, strict=True))

        results = tuple(
            RequestOutput(
                request_id=request.request_id,
                num_input_tokens_computed=len(request.input_token_ids),
                output_token_ids=(sampled[row],) if row in sampled else (),
            )
            for row, request in enumerate(batch.requests)
        )
        return ExecutionOutput(requests=results)

    def _get_runtime(self) -> tuple[PagedKVCache, ModelSession]:
        with self._lock:
            if self._cache is None or self._model is None:
                raise ExecutionNotReadyError("initialize the worker before executing")
            return self._cache, self._model
