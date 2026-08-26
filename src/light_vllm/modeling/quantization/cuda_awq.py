"""AutoAWQ GEMM 布局的 FP16 CUDA 后端。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch
from torch import Tensor

_PACK_FACTOR = 8
_DIRECT_MAX_ROWS = 255
_SPLIT_K = 8


def cuda_awq_available() -> bool:
    """只判断运行平台；编译工具链在真正选择该 Scheme 时再验证。"""

    return torch.cuda.is_available()


@lru_cache(maxsize=1)
def _extension() -> ModuleType:
    """编译并缓存小型 CUDA 扩展。

    torch 的 extension cache 自带跨进程文件锁；TP Rank 同时启动时只会有一个
    进程实际编译，其他 Rank 等待同一份产物。
    """

    try:
        from torch.utils.cpp_extension import load
    except ImportError as exc:
        raise RuntimeError("PyTorch C++ extension support is unavailable") from exc
    source = Path(__file__).parent / "csrc" / "awq"
    try:
        return load(
            name="light_vllm_awq_cuda",
            sources=[str(source / "bindings.cpp"), str(source / "gemm_kernels.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=os.getenv("LIGHT_VLLM_EXTENSION_VERBOSE", "0") == "1",
        )
    except Exception as exc:
        raise RuntimeError(
            "CUDA AWQ requires a working C++ compiler, Ninja, and CUDA toolkit; "
            "use backend=torch only for correctness debugging"
        ) from exc


def _validate_static_weights(
    qweight: Tensor,
    qzeros: Tensor,
    scales: Tensor,
    group_size: int,
) -> tuple[int, int]:
    if qweight.device.type != "cuda":
        raise ValueError("CUDA AWQ weights must be on CUDA")
    if torch.cuda.get_device_capability(qweight.device) < (7, 5):
        raise ValueError("CUDA AWQ requires compute capability 7.5 or newer")
    if qweight.dtype != torch.int32 or qzeros.dtype != torch.int32:
        raise ValueError("CUDA AWQ packed weights and zeros must use int32")
    if scales.dtype != torch.float16:
        raise ValueError("CUDA AWQ first-stage scales must use float16")
    if any(tensor.device != qweight.device for tensor in (qzeros, scales)):
        raise ValueError("CUDA AWQ tensors must share one CUDA device")
    if not all(tensor.is_contiguous() for tensor in (qweight, qzeros, scales)):
        raise ValueError("CUDA AWQ checkpoint tensors must be contiguous")
    inner = qweight.shape[0]
    output = qweight.shape[1] * _PACK_FACTOR
    if group_size <= 0 or inner % group_size or group_size % 32:
        raise ValueError("CUDA AWQ group_size must divide K and be a multiple of 32")
    if output % 64:
        raise ValueError("CUDA AWQ output width must be divisible by 64")
    expected = (inner // group_size, output // _PACK_FACTOR)
    if qzeros.shape != expected or scales.shape != (expected[0], output):
        raise ValueError("CUDA AWQ qzeros/scales do not match qweight and group_size")
    return inner, output


@dataclass(frozen=True, slots=True)
class CudaAWQOperation:
    """直接使用唯一一份 AutoAWQ INT4 权重的已准备算子。"""

    module: ModuleType
    qweight: Tensor
    qzeros: Tensor
    scales: Tensor
    group_size: int
    bias: Tensor | None
    inner: int
    output_width: int

    def __call__(self, inputs: Tensor) -> Tensor:
        # Model 构造和 prepare 已固定 dtype/device/宽度；热路径与 Dense 一样直接
        # 计算，避免每层每 token 重复查询 CUDA device capability 和 tensor 元数据。
        flattened = inputs.reshape(-1, self.inner).contiguous()
        if flattened.shape[0] > _DIRECT_MAX_ROWS:
            # 大 M 时临时展开一次权重，让 cuBLAS 充分利用更大的矩阵。模型中仍
            # 只常驻 INT4 checkpoint，不保留第二份 FP16 权重。
            weight = self.module.awq_dequantize(
                self.qweight,
                self.scales,
                self.qzeros,
                0,
                0,
                0,
            )
            output = torch.matmul(flattened, weight)
        else:
            output = self.module.awq_gemm(
                flattened,
                self.qweight,
                self.scales,
                self.qzeros,
                _SPLIT_K,
            )
        if self.bias is not None:
            output.add_(self.bias)
        return output.reshape(inputs.shape[:-1] + (self.output_width,))


def prepare_cuda_awq_operation(
    qweight: Tensor,
    qzeros: Tensor,
    scales: Tensor,
    group_size: int,
    bias: Tensor | None,
) -> CudaAWQOperation:
    inner, output = _validate_static_weights(qweight, qzeros, scales, group_size)
    if bias is not None and (
        bias.device != qweight.device or bias.dtype != scales.dtype or bias.shape != (output,)
    ):
        raise ValueError("CUDA AWQ bias must match output width, dtype, and device")
    return CudaAWQOperation(
        _extension(),
        qweight,
        qzeros,
        scales,
        group_size,
        bias,
        inner,
        output,
    )


def cuda_awq_linear(
    inputs: Tensor,
    qweight: Tensor,
    qzeros: Tensor,
    scales: Tensor,
    group_size: int,
    bias: Tensor | None,
) -> Tensor:
    """便于测试和 microbenchmark 的一次性入口。"""

    return prepare_cuda_awq_operation(qweight, qzeros, scales, group_size, bias)(inputs)
