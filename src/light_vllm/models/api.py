from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """加载模型时需要的配置。"""

    architecture: str
    loader: str = "init"
    model_args: Mapping[str, object] = field(default_factory=dict)
    weights: Path | None = None
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32


@dataclass(frozen=True, slots=True)
class ForwardBatch:
    """传给模型的一批右侧补齐的输入 token。

    ``input_ids`` 的形状固定为 ``[batch, padded_sequence]``；
    ``sequence_lengths`` 记录每行补齐前的有效长度。单请求或等长批次可以
    省略长度，此时默认每一行都使用完整宽度。
    """

    input_ids: Tensor
    sequence_lengths: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        batch_size, sequence_width = self.input_ids.shape
        lengths = self.sequence_lengths
        # 在契约边界统一归一化，后续模型和执行器不需要处理 None。
        lengths = (sequence_width,) * batch_size if lengths is None else tuple(lengths)

        if len(lengths) != batch_size:
            raise ValueError("sequence_lengths must contain one value per batch row")
        if any(
            type(length) is not int or length <= 0 or length > sequence_width for length in lengths
        ):
            raise ValueError("sequence lengths must be within the padded sequence width")
        object.__setattr__(self, "sequence_lengths", lengths)


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """模型一次计算的输出。"""

    logits: Tensor


class ModelNotLoadedError(RuntimeError):
    """还没加载模型就调用推理时抛出。"""


class ModelFactory(Protocol):
    """根据模型配置创建一个模型。"""

    def __call__(self, spec: ModelSpec) -> nn.Module: ...


class ModelForwarder(Protocol):
    """执行层调用模型运行时所需的最小接口。"""

    @property
    def generation(self) -> int: ...

    def forward(self, batch: ForwardBatch) -> ModelOutput: ...
