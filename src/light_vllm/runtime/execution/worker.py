"""进程内模型 Worker 实现。"""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock, RLock

import torch

from light_vllm.modeling.models.interfaces import (
    ForwardBatch,
    ModelNotLoadedError,
    ModelOutput,
    ModelSession,
    ModelSessionProvider,
)
from light_vllm.runtime.execution.dense_attention import (
    DenseAttentionMetadata,
    TorchDenseAttention,
)
from light_vllm.runtime.execution.interfaces import (
    DecodeHandler,
    ExecutionBatch,
    ExecutionCapabilities,
    ExecutionError,
    ExecutionLease,
    ExecutionNotReadyError,
    ExecutionOutput,
    ModelStepHandler,
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


class ContiguousStepHandler:
    """使用请求级连续 KV tensor 执行模型，作为正确性基线。"""

    def __init__(self, model: ModelSession, cache_config: ContiguousKVCacheConfig) -> None:
        model_spec = model.kv_cache_spec
        if model_spec is None:
            raise ExecutionNotReadyError("load a cacheable model before initializing the worker")
        self._cache = ContiguousKVCache(model_spec, cache_config)
        self._device = cache_config.device

    @property
    def max_kv_cache_tokens(self) -> int | None:
        return None

    def add_request(self, request_id: str, *, capacity: int) -> None:
        self._cache.allocate(request_id, capacity)

    def free_request(self, request_id: str) -> None:
        self._cache.free(request_id)

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease:
        return self._cache.acquire(request_ids)

    def forward(
        self,
        model: ModelSession,
        batch: ExecutionBatch,
    ) -> tuple[torch.Tensor, ...]:
        logits: list[torch.Tensor] = []
        for request in batch.requests:
            if request.block_ids is not None:
                raise ExecutionError("contiguous model step does not accept a block table")
            try:
                cached_tokens = self._cache.cached_tokens(request.request_id)
                if cached_tokens != request.num_computed_tokens:
                    raise ExecutionError(
                        "physical KV length must match the scheduled computed-token count"
                    )
                input_ids = torch.tensor(
                    [request.input_token_ids],
                    dtype=torch.long,
                    device=self._device,
                )
                positions = torch.arange(
                    cached_tokens,
                    cached_tokens + len(request.input_token_ids),
                    dtype=torch.long,
                    device=self._device,
                ).unsqueeze(0)
                # 连续与分页路径都向模型提供同一个 AttentionContext。
                # 区别只留在上下文如何读取和写入物理 K/V。
                attention = TorchDenseAttention(
                    self._cache.model_spec,
                    DenseAttentionMetadata(
                        positions=positions,
                        query_lengths=(len(request.input_token_ids),),
                    ),
                    self._cache.view(request.request_id),
                )
                output = _forward(
                    model,
                    ForwardBatch(
                        input_ids=input_ids,
                        positions=positions,
                        attention=attention,
                    ),
                )
                updates = attention.cache_updates
                if updates.num_tokens != len(request.input_token_ids):
                    raise ExecutionError("attention returned the wrong number of KV cache updates")
                self._cache.append(request.request_id, updates)
            except KVCacheError as exc:
                raise ExecutionError(str(exc)) from exc
            logits.append(output.logits[0])
        return tuple(logits)

    def truncate(self, request_id: str, num_cached_tokens: int) -> None:
        try:
            self._cache.truncate(request_id, num_cached_tokens)
        except KVCacheError as exc:
            raise ExecutionError(str(exc)) from exc


class _PagedExecutionLease:
    """分页缓存池不会在单次执行中销毁，所以这里无需额外保留资源。"""

    def release(self) -> None:
        return


class PagedStepHandler:
    """使用全局分页 KV cache 和同批次 Paged Attention 执行模型。"""

    def __init__(
        self,
        model: ModelSession,
        cache_planner: PagedKVCachePlanner,
        attention_backend: PagedAttentionBackend,
    ) -> None:
        model_spec = model.kv_cache_spec
        if model_spec is None:
            raise ExecutionNotReadyError("load a cacheable model before initializing the worker")
        cache_config = cache_planner.plan(model_spec)
        self._cache = PagedKVCache(model_spec, cache_config)
        self._attention_backend = attention_backend
        self._device = cache_config.device

    @property
    def max_kv_cache_tokens(self) -> int:
        return self._cache.config.num_blocks * self._cache.config.block_size

    def add_request(self, request_id: str, *, capacity: int) -> None:
        # 页表由 Scheduler 每轮传入，这里没有请求级页对象需要创建。
        if not request_id:
            raise ValueError("request_id must not be empty")
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")

    def free_request(self, request_id: str) -> None:
        # Scheduler 负责归还 page ID；全局物理页池随 Handler 一起销毁。
        return

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease:
        return _PagedExecutionLease()

    def forward(
        self,
        model: ModelSession,
        batch: ExecutionBatch,
    ) -> tuple[torch.Tensor, ...]:
        for request in batch.requests:
            if request.block_ids is None:
                raise ExecutionError("paged model step requires a block table for every request")

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
        # Handler 选择具体 attention 实现，模型只通过统一接口调用它。
        attention = self._attention_backend.create(self._cache, metadata)
        output = _forward(
            model,
            ForwardBatch(
                input_ids=input_ids,
                positions=positions,
                sequence_lengths=tuple(query_lengths),
                attention=attention,
            ),
        )
        expected_layers = frozenset(layer.layer_id for layer in self._cache.model_spec.layers)
        if attention.layer_ids != expected_layers:
            raise ExecutionError("model did not execute every configured paged attention layer")
        return tuple(
            output.logits[row, :query_length] for row, query_length in enumerate(query_lengths)
        )

    def truncate(self, request_id: str, num_cached_tokens: int) -> None:
        # 分页页池没有请求级有效长度；被拒绝的位置会在下次写入时覆盖。
        return


class StandardDecodeHandler:
    """执行一次模型步骤；只有到输出边界时才采样一个新 token。"""

    def __init__(self, sampler: Sampler) -> None:
        self._sampler = sampler

    def execute(
        self,
        model: ModelSession,
        batch: ExecutionBatch,
        step: ModelStepHandler,
    ) -> ExecutionOutput:
        for request in batch.requests:
            if request.num_lookahead_tokens:
                raise ExecutionError("standard decode does not consume lookahead tokens")
            if request.max_output_tokens > 1:
                raise ExecutionError("standard decode returns at most one output token")

        logits_by_request = step.forward(model, batch)
        if len(logits_by_request) != len(batch.requests):
            raise ExecutionError("model step must return one logits tensor per request")
        for request, logits in zip(batch.requests, logits_by_request, strict=True):
            if logits.ndim != 2 or logits.shape[0] != len(request.input_token_ids):
                raise ExecutionError("model step logits must have shape [query, vocabulary]")

        sampling_rows = [
            row for row, request in enumerate(batch.requests) if request.max_output_tokens
        ]
        sampled: dict[int, int] = {}
        if sampling_rows:
            sample_logits: list[torch.Tensor] = []
            for row in sampling_rows:
                logits = logits_by_request[row]
                sample_logits.append(logits[-1])
            sampled_ids = self._sampler.sample(torch.stack(sample_logits))
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


class LocalModelWorker:
    """固定模型版本，并把物理执行和解码语义交给注入的 Handler。"""

    def __init__(
        self,
        runner: ModelSessionProvider,
        step_factory: Callable[[ModelSession], ModelStepHandler],
        decode_handler: DecodeHandler,
    ) -> None:
        self._runner = runner
        self._step_factory = step_factory
        self._decode_handler = decode_handler
        self._model: ModelSession | None = None
        self._step: ModelStepHandler | None = None
        self._active_requests: set[str] = set()
        self._lock = RLock()
        self._initialize_lock = Lock()

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._is_ready_locked()

    @property
    def capabilities(self) -> ExecutionCapabilities:
        with self._lock:
            model, step = self._get_runtime_locked()
            return ExecutionCapabilities(
                max_model_tokens=model.max_model_tokens,
                max_kv_cache_tokens=step.max_kv_cache_tokens,
            )

    def initialize(self) -> None:
        with self._initialize_lock:
            with self._lock:
                if self._active_requests:
                    raise ExecutionError("cannot initialize worker with active requests")
            try:
                model = self._runner.open_session()
            except ModelNotLoadedError as exc:
                raise ExecutionNotReadyError("load a model before initializing the worker") from exc

            with self._lock:
                if self._is_model_installed_locked(model.generation):
                    return

            # 物理缓存可能很大；先完整构造候选对象，再一次性替换运行状态。
            candidate_step = self._step_factory(model)
            with self._lock:
                if self._active_requests:
                    raise ExecutionError("cannot initialize worker with active requests")
                if self._runner.generation != model.generation:
                    raise ExecutionError("model changed while initializing the worker")
                if self._is_model_installed_locked(model.generation):
                    return
                self._model = model
                self._step = candidate_step

    def add_request(self, request_id: str, *, capacity: int) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        with self._lock:
            if not self._is_ready_locked():
                raise ExecutionNotReadyError("initialize the worker before adding requests")
            if request_id in self._active_requests:
                raise ExecutionError(f"request {request_id!r} already exists in the worker")
            _, step = self._get_runtime_locked()
            step.add_request(request_id, capacity=capacity)
            self._active_requests.add(request_id)

    def free_request(self, request_id: str) -> bool:
        with self._lock:
            if request_id not in self._active_requests:
                return False
            _, step = self._get_runtime_locked()
            step.free_request(request_id)
            self._active_requests.remove(request_id)
            return True

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease:
        request_ids = tuple(request_ids)
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("reserved request IDs must be unique")
        with self._lock:
            _, step = self._get_runtime_locked()
            missing = tuple(
                request_id for request_id in request_ids if request_id not in self._active_requests
            )
            if missing:
                raise ExecutionError(f"worker requests are not active: {missing!r}")
            return step.acquire(request_ids)

    def execute(self, batch: ExecutionBatch) -> ExecutionOutput:
        with self._lock:
            model, step = self._get_runtime_locked()
            missing = tuple(
                request_id
                for request_id in batch.request_ids
                if request_id not in self._active_requests
            )
            if missing:
                raise ExecutionError(f"worker requests are not active: {missing!r}")
        return self._decode_handler.execute(model, batch, step)

    def _is_ready_locked(self) -> bool:
        return (
            self._model is not None
            and self._step is not None
            and self._model.generation == self._runner.generation
        )

    def _is_model_installed_locked(self, generation: int) -> bool:
        return (
            self._model is not None
            and self._step is not None
            and self._model.generation == generation
            and self._runner.generation == generation
        )

    def _get_runtime_locked(self) -> tuple[ModelSession, ModelStepHandler]:
        if self._model is None or self._step is None:
            raise ExecutionNotReadyError("initialize the worker before executing")
        return self._model, self._step
