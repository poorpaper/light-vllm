"""在调用方当前 CUDA stream 上执行模型 NCCL collective。"""

from __future__ import annotations

import ctypes
from typing import Any, Protocol

import torch
import torch.distributed as dist
from torch import Tensor

from light_vllm.runtime.execution.interfaces import ExecutionError


class _NCCLUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


_NCCL_DTYPE_BY_TORCH = {
    torch.int8: 0,
    torch.uint8: 1,
    torch.int32: 2,
    torch.int64: 4,
    torch.float16: 6,
    torch.float32: 7,
    torch.float64: 8,
    torch.bfloat16: 9,
}


class _NCCLCollectiveBindings(Protocol):
    def all_reduce_sum(self, tensor: Tensor, communicator: object, stream: int) -> None: ...

    def all_gather(
        self,
        tensor: Tensor,
        output: Tensor,
        communicator: object,
        stream: int,
    ) -> None: ...

    def destroy(self, communicator: object) -> None: ...


class _NCCLLibrary:
    """只绑定模型热路径需要的 NCCL C API，避免引入额外运行时依赖。"""

    def __init__(self) -> None:
        try:
            self._library = ctypes.CDLL("libnccl.so.2")
        except OSError as exc:
            raise ExecutionError("failed to load libnccl.so.2") from exc

        self._get_error_string = self._bind("ncclGetErrorString", ctypes.c_char_p, [ctypes.c_int])
        self._get_unique_id = self._bind(
            "ncclGetUniqueId", ctypes.c_int, [ctypes.POINTER(_NCCLUniqueId)]
        )
        self._comm_init_rank = self._bind(
            "ncclCommInitRank",
            ctypes.c_int,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _NCCLUniqueId, ctypes.c_int],
        )
        self._all_reduce = self._bind(
            "ncclAllReduce",
            ctypes.c_int,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ],
        )
        self._all_gather = self._bind(
            "ncclAllGather",
            ctypes.c_int,
            [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ],
        )
        self._comm_destroy = self._bind("ncclCommDestroy", ctypes.c_int, [ctypes.c_void_p])

    def _bind(
        self,
        name: str,
        restype: object,
        argtypes: list[object],
    ) -> Any:
        try:
            function = getattr(self._library, name)
        except AttributeError as exc:
            raise ExecutionError(f"NCCL library does not expose {name}") from exc
        function.restype = restype
        function.argtypes = argtypes
        return function

    def _check(self, result: int, operation: str) -> None:
        if result == 0:
            return
        details = self._get_error_string(result)
        message = details.decode("utf-8", errors="replace") if details else "unknown error"
        raise ExecutionError(f"{operation} failed: {message}")

    def get_unique_id(self) -> bytes:
        unique_id = _NCCLUniqueId()
        self._check(self._get_unique_id(ctypes.byref(unique_id)), "ncclGetUniqueId")
        return ctypes.string_at(ctypes.addressof(unique_id), ctypes.sizeof(unique_id))

    def init_rank(self, world_size: int, unique_id: bytes, rank: int) -> object:
        if len(unique_id) != ctypes.sizeof(_NCCLUniqueId):
            raise ExecutionError("current-stream NCCL communicator ID has an invalid size")
        communicator_id = _NCCLUniqueId.from_buffer_copy(unique_id)
        communicator = ctypes.c_void_p()
        self._check(
            self._comm_init_rank(ctypes.byref(communicator), world_size, communicator_id, rank),
            "ncclCommInitRank",
        )
        return communicator

    def all_reduce_sum(self, tensor: Tensor, communicator: object, stream: int) -> None:
        self._check(
            self._all_reduce(
                ctypes.c_void_p(tensor.data_ptr()),
                ctypes.c_void_p(tensor.data_ptr()),
                tensor.numel(),
                self._data_type(tensor.dtype),
                0,
                communicator,
                ctypes.c_void_p(stream),
            ),
            "ncclAllReduce",
        )

    def all_gather(
        self,
        tensor: Tensor,
        output: Tensor,
        communicator: object,
        stream: int,
    ) -> None:
        self._check(
            self._all_gather(
                ctypes.c_void_p(tensor.data_ptr()),
                ctypes.c_void_p(output.data_ptr()),
                tensor.numel(),
                self._data_type(tensor.dtype),
                communicator,
                ctypes.c_void_p(stream),
            ),
            "ncclAllGather",
        )

    def destroy(self, communicator: object) -> None:
        self._check(self._comm_destroy(communicator), "ncclCommDestroy")

    @staticmethod
    def _data_type(dtype: torch.dtype) -> int:
        try:
            return _NCCL_DTYPE_BY_TORCH[dtype]
        except KeyError as exc:
            raise ExecutionError(f"NCCL tensor collective does not support {dtype}") from exc


class CurrentStreamNCCL:
    """用一个 communicator 保持模型计算与 collective 的自然 stream 依赖。"""

    def __init__(
        self,
        library: _NCCLCollectiveBindings,
        communicator: object,
        world_size: int,
    ) -> None:
        self._library = library
        self._communicator: object | None = communicator
        self._world_size = world_size

    def all_reduce_sum(self, tensor: Tensor) -> Tensor:
        if not tensor.is_contiguous():
            raise ExecutionError("current-stream NCCL all-reduce requires contiguous input")
        communicator = self._ensure_open()
        stream = torch.cuda.current_stream(tensor.device)
        self._library.all_reduce_sum(tensor, communicator, stream.cuda_stream)
        return tensor

    def all_gather(self, tensor: Tensor) -> Tensor:
        if not tensor.is_contiguous():
            raise ExecutionError("current-stream NCCL all-gather requires contiguous input")
        communicator = self._ensure_open()
        gathered = tensor.new_empty(self._world_size, *tensor.shape)
        stream = torch.cuda.current_stream(tensor.device)
        self._library.all_gather(tensor, gathered, communicator, stream.cuda_stream)
        return gathered

    def close(self) -> None:
        communicator = self._communicator
        if communicator is None:
            return
        self._communicator = None
        self._library.destroy(communicator)

    def _ensure_open(self) -> object:
        if self._communicator is None:
            raise ExecutionError("current-stream NCCL communicator is closed")
        return self._communicator


def create_current_stream_nccl(
    *,
    rank: int,
    world_size: int,
    control_group: dist.ProcessGroup,
) -> CurrentStreamNCCL:
    """通过 CPU control group 交换 NCCL ID，并创建每 Rank 一个 communicator。"""

    library = _NCCLLibrary()
    values = [library.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(values, src=0, group=control_group)
    if not isinstance(values[0], bytes):
        raise ExecutionError("current-stream NCCL communicator ID is missing")
    communicator = library.init_rank(world_size, values[0], rank)
    return CurrentStreamNCCL(library, communicator, world_size)
