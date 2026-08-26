"""量化实现和模型结构之间的稳定契约。

量化只改变 Linear 的权重表示和矩阵乘实现，不应扩散到 Engine、Scheduler、
Worker 或 HTTP。模型也不识别 AWQ、FP8 等格式名，只在构造时请求列并行、
行并行层，并在权重加载完成后取得已经准备好的本地算子。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from torch import Tensor, nn

from light_vllm.modeling.tensor_parallel import TensorParallelContext, TensorPartition

if TYPE_CHECKING:
    from light_vllm.modeling.models.interfaces import ModelSpec


class LinearOperation(Protocol):
    """一个已经完成权重准备、只计算本 Rank 输出的线性算子。"""

    def __call__(self, inputs: Tensor) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class DirectLinear:
    """可以由模型直接发起 ``mm/addmm`` 的 Dense 权重。

    Dense 使用这个显式结果保留原有热路径，不需要每层经过 Python 策略对象。
    量化实现则返回 ``LinearOperation``，在构造期而不是每个 token 的热路径选择
    对应 kernel。
    """

    weight_t: Tensor
    bias: Tensor | None


PreparedLinear = DirectLinear | LinearOperation


class QuantizationMethodFactory(Protocol):
    """把 checkpoint 量化元数据解析成模型构造所需的 LinearMethod。"""

    def __call__(
        self,
        config: Mapping[str, object],
        spec: ModelSpec,
    ) -> LinearMethod: ...


class ColumnParallelLayer(Protocol):
    """Qwen 需要的列并行 Linear 最小能力。"""

    def __call__(self, inputs: Tensor) -> Tensor: ...

    def gather_output(self, local_output: Tensor) -> Tensor: ...


class RowParallelLayer(Protocol):
    """Qwen 需要的行并行 Linear 最小能力。"""

    def __call__(self, inputs: Tensor) -> Tensor: ...

    def reduce_output(self, local_output: Tensor) -> Tensor: ...


class LinearMethod(Protocol):
    """创建和准备模型 Linear 的可替换策略。

    ``create_*`` 决定参数如何保存在 ``state_dict`` 中；``prepare_*`` 决定推理时
    使用直接 Dense GEMM 还是打包量化 kernel。两个阶段由同一对象负责，避免
    loader 根据具体模型参数名维护条件树。
    """

    def create_column(
        self,
        in_features: int,
        out_features: int,
        parallel: TensorParallelContext,
        *,
        prefix: str,
        bias: bool = True,
        gather_output: bool = False,
        output_partition: TensorPartition | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Module: ...

    def create_row(
        self,
        in_features: int,
        out_features: int,
        parallel: TensorParallelContext,
        *,
        prefix: str,
        bias: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> nn.Module: ...

    def prepare_local(self, layer: nn.Module) -> PreparedLinear: ...

    def prepare_merged(self, layers: Sequence[nn.Module]) -> PreparedLinear: ...
