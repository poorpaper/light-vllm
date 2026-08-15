from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor, nn

from light_vllm.modeling.attention.interfaces import AttentionContext, ModelKVCacheSpec


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
    """传给模型的一批输入 token。

    ``input_ids`` 的形状固定为 ``[batch, padded_sequence]``；
    ``positions`` 使用相同形状，记录每个 token 在请求中的绝对位置；
    ``sequence_lengths`` 记录每行补齐前的有效长度。单请求或等长批次可以
    省略 positions 和长度，此时 positions 从零开始、每一行都使用完整宽度。
    可缓存 attention 模型必须调用 ``attention``；具体连续或分页布局由执行端注入。
    不含 attention 的模型可以省略这个字段。
    """

    input_ids: Tensor
    positions: Tensor | None = None
    sequence_lengths: tuple[int, ...] | None = None
    attention: AttentionContext | None = None

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        batch_size, sequence_width = self.input_ids.shape
        positions = self.positions
        if positions is None:
            positions = torch.arange(
                sequence_width,
                dtype=torch.long,
                device=self.input_ids.device,
            ).expand(batch_size, -1)
        if positions.shape != self.input_ids.shape:
            raise ValueError("positions must have the same shape as input_ids")
        if positions.dtype != torch.long or positions.device != self.input_ids.device:
            raise ValueError("positions must use torch.long on the input_ids device")
        if bool(torch.any(positions < 0)):
            raise ValueError("positions must not be negative")

        lengths = self.sequence_lengths
        # 在契约边界统一归一化，后续模型和执行器不需要处理 None。
        lengths = (sequence_width,) * batch_size if lengths is None else tuple(lengths)

        if len(lengths) != batch_size:
            raise ValueError("sequence_lengths must contain one value per batch row")
        if any(
            type(length) is not int or length <= 0 or length > sequence_width for length in lengths
        ):
            raise ValueError("sequence lengths must be within the padded sequence width")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "sequence_lengths", lengths)


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """模型一次计算得到的 logits。"""

    logits: Tensor


class ModelNotLoadedError(RuntimeError):
    """还没加载模型就调用推理时抛出。"""


class ModelFactory(Protocol):
    """根据模型配置创建一个模型。"""

    def __call__(self, spec: ModelSpec) -> nn.Module: ...


class ModelSession(Protocol):
    """一次生成或一个 Worker 选定后持续使用的模型。

    这里的“固定”只表示执行途中不会切换到后来重新加载的模型，并不表示
    ``nn.Module`` 本身是不可修改的对象。
    """

    @property
    def generation(self) -> int: ...

    @property
    def kv_cache_spec(self) -> ModelKVCacheSpec | None: ...

    @property
    def max_model_tokens(self) -> int | None: ...

    def forward(self, batch: ForwardBatch) -> ModelOutput: ...


class ModelSessionProvider(Protocol):
    """让执行代码取得当前模型，并在本次执行期间继续使用它。"""

    @property
    def generation(self) -> int: ...

    def open_session(self) -> ModelSession: ...
