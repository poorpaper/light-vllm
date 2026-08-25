"""基于 torch.distributed 的单机 Tensor Parallel 执行拓扑。"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import timedelta
from math import isfinite
from threading import Lock

import torch
import torch.distributed as dist
from torch import Tensor

from light_vllm.modeling.attention.interfaces import ModelKVCacheSpec
from light_vllm.modeling.tensor_parallel import TensorCollectives
from light_vllm.runtime.execution.interfaces import (
    ExecutionBatch,
    ExecutionCapabilities,
    ExecutionError,
    ExecutionLease,
    ExecutionOutput,
    ExecutionTimer,
    ModelExecutor,
)
from light_vllm.runtime.execution.paged_cache import (
    PagedKVCacheConfig,
    PagedKVCachePlanner,
)


class TorchDistributedGroup(TensorCollectives):
    """同时提供模型 tensor collective 与轻量 CPU 控制通信。"""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        local_rank: int,
        device: torch.device,
        control_group: dist.ProcessGroup,
        tensor_group: dist.ProcessGroup,
        owns_world_group: bool,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = device
        self._control_group = control_group
        self._tensor_group = tensor_group
        self._owns_world_group = owns_world_group
        self._closed = False

    @classmethod
    def initialize(
        cls,
        *,
        backend: str = "nccl",
        timeout_seconds: float = 120.0,
    ) -> TorchDistributedGroup:
        """按 torchrun 环境变量初始化一组 Rank。"""

        if not torch.distributed.is_available():
            raise ExecutionError("this PyTorch build does not provide distributed execution")
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("distributed timeout must be positive")
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
        except (KeyError, ValueError) as exc:
            raise ExecutionError("tensor parallel serving must be launched with torchrun") from exc
        if world_size <= 1:
            raise ExecutionError("distributed tensor parallelism requires at least two ranks")
        if local_world_size != world_size:
            raise ExecutionError("this tensor parallel executor currently supports one node only")
        if not 0 <= rank < world_size or not 0 <= local_rank < local_world_size:
            raise ExecutionError("torchrun rank is outside the configured world size")
        if backend == "nccl":
            if not torch.cuda.is_available():
                raise ExecutionError("NCCL tensor parallelism requires CUDA")
            if not 0 <= local_rank < torch.cuda.device_count():
                raise ExecutionError("LOCAL_RANK does not map to a visible CUDA device")
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        elif backend == "gloo":
            device = torch.device("cpu")
        else:
            raise ValueError(f"unsupported distributed backend: {backend}")

        timeout = timedelta(seconds=timeout_seconds)
        owns_world_group = not dist.is_initialized()
        if not owns_world_group:
            if dist.get_rank() != rank or dist.get_world_size() != world_size:
                raise ExecutionError("existing process group does not match torchrun ranks")
            if str(dist.get_backend()) != backend:
                raise ExecutionError("existing process group uses a different backend")
        else:
            options = None
            if backend == "nccl":
                # TP 的短 AllReduce 位于每层关键路径上，使用高优先级 NCCL
                # stream，避免排在同 Rank 的普通计算 stream 后面。
                options = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
                options._timeout = timeout
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timeout,
                pg_options=options,
            )
        # NCCL 只处理 device tensor；控制命令使用独立 Gloo group，避免把
        # Python 对象序列化后再绕到 GPU。vLLM/SGLang 也采用双 group 边界。
        try:
            control_group = (
                dist.new_group(backend="gloo", timeout=timeout)
                if backend == "nccl"
                else dist.group.WORLD
            )
        except Exception:
            if owns_world_group and dist.is_initialized():
                dist.destroy_process_group()
            raise
        return cls(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            control_group=control_group,
            tensor_group=dist.group.WORLD,
            owns_world_group=owns_world_group,
        )

    def all_reduce_sum(self, tensor: Tensor) -> Tensor:
        self._ensure_open()
        if tensor.device != self.device:
            raise ExecutionError("tensor collective received a tensor on the wrong device")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self._tensor_group)
        return tensor

    def all_gather_last_dim(
        self,
        tensor: Tensor,
        partition_sizes: tuple[int, ...],
    ) -> Tensor:
        self._ensure_open()
        if tensor.device != self.device:
            raise ExecutionError("tensor collective received a tensor on the wrong device")
        if len(partition_sizes) != self.world_size:
            raise ValueError("gather partitions must contain one size per rank")
        if tensor.shape[-1] != partition_sizes[self.rank]:
            raise ValueError("local tensor width does not match its gather partition")

        max_size = max(partition_sizes)
        if tensor.shape[-1] < max_size:
            padding = tensor.new_zeros(*tensor.shape[:-1], max_size - tensor.shape[-1])
            tensor = torch.cat((tensor, padding), dim=-1)
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor, group=self._tensor_group)
        return torch.cat(
            tuple(value[..., :size] for value, size in zip(gathered, partition_sizes, strict=True)),
            dim=-1,
        )

    def minimum(self, value: int) -> int:
        self._ensure_open()
        tensor = torch.tensor(value, dtype=torch.long, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=self._tensor_group)
        return int(tensor.item())

    def broadcast_object(self, value: object | None, *, src: int = 0) -> object:
        self._ensure_open()
        if not 0 <= src < self.world_size:
            raise ValueError("broadcast source rank is outside the process group")
        values = [value]
        dist.broadcast_object_list(values, src=src, group=self._control_group)
        return values[0]

    def first_rank(self, selected: bool) -> int | None:
        """返回第一个满足条件的 Rank；成功路径只传递一个整数。"""

        self._ensure_open()
        value = self.rank if selected else self.world_size
        tensor = torch.tensor(value, dtype=torch.long)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=self._control_group)
        result = int(tensor.item())
        return None if result == self.world_size else result

    def all_gather_object(self, value: object) -> tuple[object, ...]:
        self._ensure_open()
        values: list[object | None] = [None] * self.world_size
        dist.all_gather_object(values, value, group=self._control_group)
        return tuple(values)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._control_group is not dist.group.WORLD:
            dist.destroy_process_group(self._control_group)
        if self._owns_world_group and dist.is_initialized():
            dist.destroy_process_group()

    def _ensure_open(self) -> None:
        if self._closed or not dist.is_initialized():
            raise ExecutionError("distributed process group is closed")


class SynchronizedPagedKVCachePlanner:
    """各 Rank 先独立估算显存，再统一采用最小物理页数。"""

    def __init__(
        self,
        local: PagedKVCachePlanner,
        group: TorchDistributedGroup,
    ) -> None:
        self._local = local
        self._group = group
        self._config: PagedKVCacheConfig | None = None

    @property
    def num_blocks(self) -> int:
        if self._config is None:
            raise RuntimeError("plan the distributed KV cache before reading its capacity")
        return self._config.num_blocks

    @property
    def block_size(self) -> int:
        return self._local.block_size

    @property
    def device(self) -> torch.device:
        return self._local.device

    def plan(self, model_spec: ModelKVCacheSpec) -> PagedKVCacheConfig:
        local = self._local.plan(model_spec)
        num_blocks = self._group.minimum(local.num_blocks)
        config = replace(local, num_blocks=num_blocks)
        if self._config is not None and self._config != config:
            raise RuntimeError("distributed KV cache capacity changed after initialization")
        self._config = config
        return config


@dataclass(frozen=True, slots=True)
class _Initialize:
    pass


@dataclass(frozen=True, slots=True)
class _AddRequest:
    request_id: str
    capacity: int


@dataclass(frozen=True, slots=True)
class _FreeRequest:
    request_id: str


@dataclass(frozen=True, slots=True)
class _UpdateRequests:
    updates: tuple[_AddRequest | _FreeRequest, ...]


@dataclass(frozen=True, slots=True)
class _Execute:
    batch: ExecutionBatch
    updates: tuple[_AddRequest | _FreeRequest, ...] = ()


@dataclass(frozen=True, slots=True)
class _Shutdown:
    pass


_Command = _Initialize | _UpdateRequests | _Execute | _Shutdown


@dataclass(frozen=True, slots=True)
class _CommandResult:
    value: object | None = None
    error_type: str | None = None
    error_message: str | None = None


def _apply_command(executor: ModelExecutor, command: _Command) -> object | None:
    if isinstance(command, _Initialize):
        executor.initialize()
        return executor.capabilities
    if isinstance(command, _UpdateRequests):
        _apply_request_updates(executor, command.updates)
        return None
    if isinstance(command, _Execute):
        _apply_request_updates(executor, command.updates)
        return executor.execute(command.batch)
    if isinstance(command, _Shutdown):
        return None
    raise TypeError(f"unsupported tensor-parallel command: {type(command).__name__}")


def _apply_request_updates(
    executor: ModelExecutor,
    updates: tuple[_AddRequest | _FreeRequest, ...],
) -> None:
    for update in updates:
        if isinstance(update, _AddRequest):
            executor.add_request(update.request_id, capacity=update.capacity)
        else:
            removed = executor.free_request(update.request_id)
            if not removed:
                raise ExecutionError(
                    f"tensor parallel rank does not own request {update.request_id!r}"
                )


def _command_result(operation: Callable[[], object | None]) -> _CommandResult:
    try:
        return _CommandResult(value=operation())
    except Exception as exc:
        return _CommandResult(
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def _validate_results(
    results: tuple[object, ...],
    *,
    require_consensus: bool,
) -> object | None:
    typed = tuple(result for result in results if isinstance(result, _CommandResult))
    if len(typed) != len(results):
        raise ExecutionError("tensor parallel rank returned an invalid command result")
    failures = tuple(
        (rank, result) for rank, result in enumerate(typed) if result.error_type is not None
    )
    if failures:
        details = "; ".join(
            f"rank {rank}: {result.error_type}: {result.error_message}" for rank, result in failures
        )
        raise ExecutionError(f"tensor parallel command failed: {details}")
    values = tuple(result.value for result in typed)
    if require_consensus and any(value != values[0] for value in values[1:]):
        raise ExecutionError("tensor parallel ranks returned different results")
    return values[0]


def _raise_synchronized_error(
    group: TorchDistributedGroup,
    local_result: _CommandResult,
) -> None:
    """同步执行成败；仅在失败时传输异常文本。"""

    failure_rank = group.first_rank(local_result.error_type is not None)
    if failure_rank is None:
        return
    failure = group.broadcast_object(
        local_result if group.rank == failure_rank else None,
        src=failure_rank,
    )
    if not isinstance(failure, _CommandResult) or failure.error_type is None:
        raise ExecutionError("tensor parallel rank returned an invalid failure result")
    raise ExecutionError(
        "tensor parallel command failed: "
        f"rank {failure_rank}: {failure.error_type}: {failure.error_message}"
    )


class TensorParallelModelExecutor:
    """Rank 0 的 Executor；请求状态仍由现有 EngineCore 独占。"""

    def __init__(
        self,
        local: ModelExecutor,
        group: TorchDistributedGroup,
        *,
        timer: ExecutionTimer | None = None,
    ) -> None:
        if group.rank != 0:
            raise ValueError("only rank 0 can own TensorParallelModelExecutor")
        self._local = local
        self._group = group
        self._timer = timer
        self._pending: list[_AddRequest | _FreeRequest] = []
        self._active_requests: set[str] = set()
        self._lease_counts: dict[str, int] = {}
        self._deferred_frees: set[str] = set()
        # 生命周期只用短锁更新；长时间的 collective 由另一把锁串行化。
        self._state_lock = Lock()
        self._command_lock = Lock()
        self._capabilities: ExecutionCapabilities | None = None
        self._closed = False

    @property
    def ready(self) -> bool:
        with self._state_lock:
            return not self._closed and self._capabilities is not None and self._local.ready

    @property
    def capabilities(self) -> ExecutionCapabilities:
        with self._state_lock:
            capabilities = self._capabilities
        if capabilities is None:
            raise ExecutionError("initialize the tensor parallel executor first")
        return capabilities

    def initialize(self) -> None:
        def initialize_local() -> ExecutionCapabilities:
            self._local.initialize()
            return self._local.capabilities

        with self._command_lock:
            with self._state_lock:
                self._ensure_open_locked()
            value = self._run_command(
                _Initialize(),
                initialize_local,
                require_consensus=True,
            )
            if not isinstance(value, ExecutionCapabilities):
                raise ExecutionError("tensor parallel initialization returned invalid capabilities")
            if value.tensor_parallel_size != self._group.world_size:
                raise ExecutionError(
                    "model tensor parallel size does not match the distributed world size"
                )
            with self._state_lock:
                self._capabilities = value

    def add_request(self, request_id: str, *, capacity: int) -> None:
        # Rank 0 立即登记请求；远端更新在下一条有序命令前一起送达。
        with self._state_lock:
            self._ensure_open_locked()
            self._local.add_request(request_id, capacity=capacity)
            self._active_requests.add(request_id)
            self._pending.append(_AddRequest(request_id, capacity))

    def free_request(self, request_id: str) -> bool:
        with self._state_lock:
            self._ensure_open_locked()
            if request_id not in self._active_requests:
                return False
            self._active_requests.remove(request_id)
            if self._lease_counts.get(request_id, 0):
                self._deferred_frees.add(request_id)
            else:
                self._free_local(request_id)
            return True

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease:
        with self._state_lock:
            self._ensure_open_locked()
            local = self._local.acquire(request_ids)
            for request_id in request_ids:
                self._lease_counts[request_id] = self._lease_counts.get(request_id, 0) + 1
        return _TensorParallelLease(self, local, request_ids)

    def execute(self, batch: ExecutionBatch) -> ExecutionOutput:
        def operation() -> ExecutionOutput:
            return self._execute(batch)

        if self._timer is None:
            return operation()
        output, elapsed_seconds = self._timer.measure(operation)
        return ExecutionOutput(
            requests=output.requests,
            num_model_tokens_computed=output.num_model_tokens_computed,
            step_elapsed_seconds=elapsed_seconds,
        )

    def shutdown(self) -> None:
        with self._command_lock:
            with self._state_lock:
                if self._closed:
                    return
                # 先关闭新生命周期更新，再把已有更新按序送达所有 Rank。
                self._closed = True
            self._flush_pending()
            self._run_command(_Shutdown(), lambda: None, require_consensus=True)

    def _execute(self, batch: ExecutionBatch) -> ExecutionOutput:
        with self._command_lock:
            with self._state_lock:
                self._ensure_open_locked()
                updates = self._take_pending_locked()
            value = self._run_command(
                _Execute(batch, updates),
                lambda: self._local.execute(batch),
                require_consensus=False,
            )
        if not isinstance(value, ExecutionOutput):
            raise ExecutionError("tensor parallel execution returned an invalid output")
        return value

    def _flush_pending(self) -> None:
        with self._state_lock:
            updates = self._take_pending_locked()
        if not updates:
            return
        self._run_command(
            _UpdateRequests(updates),
            lambda: None,
            require_consensus=True,
        )

    def _take_pending_locked(self) -> tuple[_AddRequest | _FreeRequest, ...]:
        updates = tuple(self._pending)
        self._pending.clear()
        return updates

    def _release_lease(self, local: ExecutionLease, request_ids: tuple[str, ...]) -> None:
        with self._state_lock:
            local.release()
            for request_id in request_ids:
                count = self._lease_counts[request_id] - 1
                if count:
                    self._lease_counts[request_id] = count
                    continue
                del self._lease_counts[request_id]
                if request_id in self._deferred_frees:
                    self._deferred_frees.remove(request_id)
                    self._free_local(request_id)

    def _free_local(self, request_id: str) -> None:
        if not self._local.free_request(request_id):
            raise ExecutionError(f"tensor parallel leader does not own request {request_id!r}")
        self._pending.append(_FreeRequest(request_id))

    def _ensure_open_locked(self) -> None:
        if self._closed:
            raise ExecutionError("tensor parallel executor is closed")

    def _run_command(
        self,
        command: _Command,
        operation: Callable[[], object | None],
        *,
        require_consensus: bool,
    ) -> object | None:
        self._group.broadcast_object(command)
        local_result = _command_result(operation)
        if not require_consensus:
            # Rank 0 独占 Engine 状态和对外输出；其他 Rank 只需保持模型与 KV
            # 状态同步。成功时不再序列化所有 Rank 的完整 ExecutionOutput。
            _raise_synchronized_error(self._group, local_result)
            return local_result.value
        results = self._group.all_gather_object(local_result)
        return _validate_results(results, require_consensus=require_consensus)


class _TensorParallelLease:
    """把 Rank 0 的本地资源和跨 Rank 请求生命周期固定到同一安全边界。"""

    def __init__(
        self,
        owner: TensorParallelModelExecutor,
        local: ExecutionLease,
        request_ids: tuple[str, ...],
    ) -> None:
        self._owner = owner
        self._local = local
        self._request_ids = request_ids
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._owner._release_lease(self._local, self._request_ids)
        self._released = True


def run_tensor_parallel_worker(
    executor: ModelExecutor,
    group: TorchDistributedGroup,
) -> None:
    """非零 Rank 的阻塞命令循环；所有模型步骤顺序与 Rank 0 一致。"""

    if group.rank == 0:
        raise ValueError("rank 0 owns the engine and cannot enter the worker loop")
    while True:
        command = group.broadcast_object(None)
        if not isinstance(command, (_Initialize, _UpdateRequests, _Execute, _Shutdown)):
            result = _CommandResult(
                error_type="TypeError",
                error_message="received an invalid tensor parallel command",
            )
        else:
            result = _command_result(lambda current=command: _apply_command(executor, current))
        if isinstance(command, _Execute):
            _raise_synchronized_error(group, result)
        else:
            results = group.all_gather_object(result)
            _validate_results(results, require_consensus=False)
        if isinstance(command, _Shutdown):
            return
