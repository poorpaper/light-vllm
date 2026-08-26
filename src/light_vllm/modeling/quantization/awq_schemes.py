"""AWQ 运行时后端选择。

所有 Scheme 消费同一份 AutoAWQ GEMM checkpoint 布局。选择只发生一次；模型
forward 持有已经准备好的 ``LinearOperation``，因此不会把 backend 分支带进
每层、每 token 的热路径。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from light_vllm.modeling.quantization.awq import _torch_awq_linear
from light_vllm.modeling.quantization.interfaces import LinearOperation


@dataclass(frozen=True, slots=True)
class TorchAWQOperation:
    """先反量化再调用 PyTorch GEMM 的正确性基线。"""

    qweight: Tensor
    qzeros: Tensor
    scales: Tensor
    group_size: int
    bias: Tensor | None

    def __call__(self, inputs: Tensor) -> Tensor:
        return _torch_awq_linear(
            inputs,
            self.qweight,
            self.qzeros,
            self.scales,
            self.group_size,
            self.bias,
        )


class TorchAWQScheme:
    """无可选依赖的 AWQ correctness Scheme。"""

    name = "torch"

    def prepare(
        self,
        qweight: Tensor,
        qzeros: Tensor,
        scales: Tensor,
        group_size: int,
        bias: Tensor | None,
    ) -> LinearOperation:
        return TorchAWQOperation(qweight, qzeros, scales, group_size, bias)


class CudaAWQScheme:
    """直接消费 AutoAWQ INT4 权重的 CUDA Scheme。"""

    name = "cuda"

    def prepare(
        self,
        qweight: Tensor,
        qzeros: Tensor,
        scales: Tensor,
        group_size: int,
        bias: Tensor | None,
    ) -> LinearOperation:
        from light_vllm.modeling.quantization.cuda_awq import (
            prepare_cuda_awq_operation,
        )

        return prepare_cuda_awq_operation(qweight, qzeros, scales, group_size, bias)


def select_awq_scheme(
    backend: str,
    *,
    device: str | torch.device,
) -> TorchAWQScheme | CudaAWQScheme:
    """在加载阶段选定唯一 Scheme，不在逐 token 热路径反复判断。"""

    if backend not in ("auto", "torch", "cuda"):
        raise ValueError("AWQ backend must be auto, torch, or cuda")
    target = torch.device(device)
    if backend == "torch":
        return TorchAWQScheme()
    if target.type != "cuda":
        if backend == "cuda":
            raise ValueError("CUDA AWQ requires a CUDA device")
        return TorchAWQScheme()
    from light_vllm.modeling.quantization.cuda_awq import cuda_awq_available

    if not cuda_awq_available():
        raise RuntimeError("CUDA AWQ backend is unavailable")
    return CudaAWQScheme()
