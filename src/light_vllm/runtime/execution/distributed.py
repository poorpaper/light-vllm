"""基于 torch.distributed 的单机 Tensor Parallel 执行拓扑。"""

from __future__ import annotations

import os
import pickle
import socket
import struct
import tempfile
from collections.abc import Callable
from contextlib import ExitStack, suppress
from dataclasses import dataclass, replace
from datetime import timedelta
from math import isfinite
from pathlib import Path
from threading import Lock
from typing import Protocol

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
from light_vllm.runtime.execution.nccl import (
    CurrentStreamNCCL,
    create_current_stream_nccl,
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
        current_stream_nccl: CurrentStreamNCCL | None,
        owns_world_group: bool,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = device
        self._control_group = control_group
        self._tensor_group = tensor_group
        self._current_stream_nccl = current_stream_nccl
        self._owns_world_group = owns_world_group
        self._command_channel: _CommandChannel | None = None
        self._closed = False

    @classmethod
    def initialize(
        cls,
        *,
        backend: str = "nccl",
        control_transport: str = "auto",
        timeout_seconds: float = 120.0,
    ) -> TorchDistributedGroup:
        """按 torchrun 环境变量初始化一组 Rank。"""

        if not torch.distributed.is_available():
            raise ExecutionError("this PyTorch build does not provide distributed execution")
        if not isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("distributed timeout must be positive")
        if control_transport not in {"auto", "socket", "gloo"}:
            raise ValueError("distributed control transport must be auto, socket, or gloo")
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
                # ProcessGroup 只保留给启动和低频容量事实；模型 collective
                # 使用独立 communicator 在模型当前 CUDA stream 上执行。
                options = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
                options._timeout = timeout
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timeout,
                pg_options=options,
            )
        # NCCL 只处理 device tensor；独立 Gloo group 负责启动协调、容量
        # 事实和可选命令回退，避免 Python 对象经 GPU collective 传输。
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
        try:
            current_stream_nccl = (
                create_current_stream_nccl(
                    rank=rank,
                    world_size=world_size,
                    control_group=control_group,
                )
                if backend == "nccl"
                else None
            )
        except Exception:
            if control_group is not dist.group.WORLD:
                dist.destroy_process_group(control_group)
            if owns_world_group and dist.is_initialized():
                dist.destroy_process_group()
            raise

        group = cls(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            control_group=control_group,
            tensor_group=dist.group.WORLD,
            current_stream_nccl=current_stream_nccl,
            owns_world_group=owns_world_group,
        )
        try:
            group._command_channel = _create_command_channel(
                group,
                transport=control_transport,
                timeout_seconds=timeout_seconds,
            )
        except Exception:
            # 初始化异常是主错误；close 已尽力释放全部已建资源。
            with suppress(Exception):
                group.close()
            raise
        return group

    @property
    def command_channel(self) -> _CommandChannel:
        channel = self._command_channel
        if channel is None:
            raise ExecutionError("tensor parallel command channel is not initialized")
        return channel

    def all_reduce_sum(self, tensor: Tensor) -> Tensor:
        self._ensure_open()
        if tensor.device != self.device:
            raise ExecutionError("tensor collective received a tensor on the wrong device")
        if self._current_stream_nccl is not None:
            return self._current_stream_nccl.all_reduce_sum(tensor)
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
        if self._current_stream_nccl is None:
            gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(gathered, tensor, group=self._tensor_group)
        else:
            gathered = self._current_stream_nccl.all_gather(tensor)
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
        command_channel = self._command_channel
        current_stream_nccl = self._current_stream_nccl
        self._command_channel = None
        self._current_stream_nccl = None

        # ExitStack 即使某个 close 失败也会继续执行其余 callback，并保留
        # command -> NCCL -> control -> world 的释放顺序。
        with ExitStack() as cleanup:
            if self._owns_world_group and dist.is_initialized():
                cleanup.callback(dist.destroy_process_group)
            if self._control_group is not dist.group.WORLD:
                cleanup.callback(dist.destroy_process_group, self._control_group)
            if current_stream_nccl is not None:
                cleanup.callback(current_stream_nccl.close)
            if command_channel is not None:
                cleanup.callback(command_channel.close)

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


class _CommandChannel(Protocol):
    """把一条命令交给所有 Rank，并把执行结果收回 Rank 0。"""

    def broadcast(self, value: object | None) -> object: ...

    def complete(
        self,
        value: object,
        *,
        failed: bool,
        require_consensus: bool,
    ) -> tuple[object, ...]: ...

    def abort(self) -> None: ...

    def close(self) -> None: ...


class _GlooCommandChannel:
    """保留原有 collective 控制通道，供显式回退和非 POSIX 环境使用。"""

    def __init__(self, group: TorchDistributedGroup) -> None:
        self._group = group

    def broadcast(self, value: object | None) -> object:
        return self._group.broadcast_object(value)

    def complete(
        self,
        value: object,
        *,
        failed: bool,
        require_consensus: bool,
    ) -> tuple[object, ...]:
        if require_consensus:
            return self._group.all_gather_object(value)

        # 正常模型 step 只同步一个成功标记；异常文本仅在失败时广播。
        failure_rank = self._group.first_rank(failed)
        if failure_rank is None:
            return (value,) if self._group.rank == 0 else ()
        failure = self._group.broadcast_object(
            value if self._group.rank == failure_rank else None,
            src=failure_rank,
        )
        return (failure,)

    def abort(self) -> None:
        return None

    def close(self) -> None:
        return None


class _SocketCommandChannel:
    """单机 Rank 间的有序 Unix socket 命令通道。

    每条 ``ExecutionBatch`` 只序列化一次；各 Worker 通过本地 socket 收取
    同一份字节，并只回传小结果。NCCL 仍只负责模型 tensor collective。
    """

    _HEADER = struct.Struct("!Q")
    _MAX_PAYLOAD_BYTES = 64 * 1024 * 1024

    def __init__(
        self,
        *,
        rank: int,
        peers: dict[int, socket.socket],
        socket_path: Path | None = None,
    ) -> None:
        self._rank = rank
        self._peers = peers
        self._socket_path = socket_path
        self._closed = False

    @classmethod
    def initialize(
        cls,
        group: TorchDistributedGroup,
        *,
        timeout_seconds: float,
    ) -> _SocketCommandChannel:
        listener: socket.socket | None = None
        socket_path: Path | None = None
        peers: dict[int, socket.socket] = {}
        try:
            if group.rank == 0:
                directory = Path(tempfile.mkdtemp(prefix="light-vllm-tp-"))
                socket_path = directory / "commands.sock"
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                listener.settimeout(timeout_seconds)
                listener.bind(str(socket_path))
                os.chmod(socket_path, 0o600)
                listener.listen(group.world_size - 1)
            address = group.broadcast_object(str(socket_path) if socket_path else None)
            if not isinstance(address, str):
                raise ExecutionError("tensor parallel command socket address is invalid")

            if group.rank == 0:
                assert listener is not None
                for _ in range(1, group.world_size):
                    peer, _ = listener.accept()
                    peer.settimeout(timeout_seconds)
                    try:
                        peer_rank = cls._receive_object(peer)
                    except Exception:
                        peer.close()
                        raise
                    if (
                        type(peer_rank) is not int
                        or not 1 <= peer_rank < group.world_size
                        or peer_rank in peers
                    ):
                        peer.close()
                        raise ExecutionError(
                            "tensor parallel command socket received an invalid rank"
                        )
                    peers[peer_rank] = peer
                return cls(rank=0, peers=peers, socket_path=socket_path)

            peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            peer.settimeout(timeout_seconds)
            peer.connect(address)
            peers[0] = peer
            cls._send_object(peer, group.rank)
            return cls(rank=group.rank, peers=peers)
        except Exception as exc:
            for peer in peers.values():
                peer.close()
            if socket_path is not None:
                cls._remove_socket_path(socket_path)
            if isinstance(exc, ExecutionError):
                raise
            raise ExecutionError("failed to initialize the local TP command socket") from exc
        finally:
            if listener is not None:
                listener.close()

    def broadcast(self, value: object | None) -> object:
        self._ensure_open()
        if self._rank == 0:
            payload = self._serialize(value)
            for rank in sorted(self._peers):
                self._send_payload(self._peers[rank], payload)
            return value
        return self._receive_object(self._peers[0])

    def complete(
        self,
        value: object,
        *,
        failed: bool,
        require_consensus: bool,
    ) -> tuple[object, ...]:
        # Worker 回包本身就是完成确认，两种上层策略都复用同一有序收集。
        del failed, require_consensus
        self._ensure_open()
        if self._rank != 0:
            self._send_object(self._peers[0], value)
            return ()
        results = [value]
        for rank in sorted(self._peers):
            results.append(self._receive_object(self._peers[rank]))
        return tuple(results)

    def abort(self) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for peer in self._peers.values():
            peer.close()
        self._peers.clear()
        if self._socket_path is not None:
            self._remove_socket_path(self._socket_path)

    def _ensure_open(self) -> None:
        if self._closed:
            raise ExecutionError("tensor parallel command socket is closed")

    @classmethod
    def _serialize(cls, value: object) -> bytes:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > cls._MAX_PAYLOAD_BYTES:
            raise ExecutionError("tensor parallel command exceeds the local IPC limit")
        return payload

    @classmethod
    def _send_object(cls, peer: socket.socket, value: object) -> None:
        cls._send_payload(peer, cls._serialize(value))

    @classmethod
    def _send_payload(cls, peer: socket.socket, payload: bytes) -> None:
        try:
            peer.sendall(cls._HEADER.pack(len(payload)) + payload)
        except OSError as exc:
            raise ExecutionError("tensor parallel command socket send failed") from exc

    @classmethod
    def _receive_object(cls, peer: socket.socket) -> object:
        try:
            header = cls._receive_exact(peer, cls._HEADER.size)
            (size,) = cls._HEADER.unpack(header)
            if size > cls._MAX_PAYLOAD_BYTES:
                raise ExecutionError("tensor parallel command socket payload is too large")
            return pickle.loads(cls._receive_exact(peer, size))
        except (OSError, pickle.PickleError, EOFError) as exc:
            raise ExecutionError("tensor parallel command socket receive failed") from exc

    @staticmethod
    def _receive_exact(peer: socket.socket, size: int) -> bytes:
        chunks = bytearray(size)
        view = memoryview(chunks)
        received = 0
        while received < size:
            count = peer.recv_into(view[received:])
            if count == 0:
                raise ExecutionError("tensor parallel command socket peer exited")
            received += count
        return bytes(chunks)

    @staticmethod
    def _remove_socket_path(socket_path: Path) -> None:
        try:
            socket_path.unlink(missing_ok=True)
            socket_path.parent.rmdir()
        except OSError:
            # 关闭阶段以释放通信资源为主；临时目录清理失败不应覆盖原异常。
            pass


def _create_command_channel(
    group: TorchDistributedGroup,
    *,
    transport: str,
    timeout_seconds: float,
) -> _CommandChannel:
    if transport == "gloo":
        return _GlooCommandChannel(group)
    supports_socket = os.name == "posix" and hasattr(socket, "AF_UNIX")
    hostnames = group.all_gather_object(socket.gethostname())
    single_host = len(set(hostnames)) == 1
    selected = "socket" if transport == "auto" and supports_socket and single_host else transport
    if selected == "auto":
        selected = "gloo"
    if selected == "gloo":
        return _GlooCommandChannel(group)
    if selected != "socket" or not supports_socket:
        raise ExecutionError("Unix socket TP control transport is unavailable on this host")
    if not single_host:
        raise ExecutionError("Unix socket TP control transport requires all ranks on one host")
    return _SocketCommandChannel.initialize(group, timeout_seconds=timeout_seconds)


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


class TensorParallelModelExecutor:
    """Rank 0 的 Executor；请求状态仍由现有 EngineCore 独占。"""

    def __init__(
        self,
        local: ModelExecutor,
        group: TorchDistributedGroup,
        *,
        command_channel: _CommandChannel | None = None,
        timer: ExecutionTimer | None = None,
    ) -> None:
        if group.rank != 0:
            raise ValueError("only rank 0 can own TensorParallelModelExecutor")
        self._local = local
        self._group = group
        self._commands = command_channel or _GlooCommandChannel(group)
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
        try:
            self._commands.broadcast(command)
            local_result = _command_result(operation)
            results = self._commands.complete(
                local_result,
                failed=local_result.error_type is not None,
                require_consensus=require_consensus,
            )
            return _validate_results(results, require_consensus=require_consensus)
        except Exception:
            # 任一 Rank 失败后，其他 Rank 的模型/KV 状态不再可安全复用。
            self._commands.abort()
            raise


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
    *,
    command_channel: _CommandChannel | None = None,
) -> None:
    """非零 Rank 的阻塞命令循环；所有模型步骤顺序与 Rank 0 一致。"""

    if group.rank == 0:
        raise ValueError("rank 0 owns the engine and cannot enter the worker loop")
    commands = command_channel or _GlooCommandChannel(group)
    while True:
        try:
            command = commands.broadcast(None)
            if not isinstance(command, (_Initialize, _UpdateRequests, _Execute, _Shutdown)):
                result = _CommandResult(
                    error_type="TypeError",
                    error_message="received an invalid tensor parallel command",
                )
            else:
                result = _command_result(lambda current=command: _apply_command(executor, current))
            require_consensus = not isinstance(command, _Execute)
            # Rank 0 独占生成结果；正常 execute 只回传小型成功/失败状态。
            wire_result = result if require_consensus else replace(result, value=None)
            results = commands.complete(
                wire_result,
                failed=result.error_type is not None,
                require_consensus=require_consensus,
            )
            _validate_results(
                results or (wire_result,),
                require_consensus=False,
            )
        except Exception:
            commands.abort()
            raise
        if isinstance(command, _Shutdown):
            return
