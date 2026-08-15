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
class LayerKeyValues:
    """一层 attention 的 key/value 张量。

    两个张量都使用 ``[batch, sequence, kv_heads, head_size]``。这里描述的是
    模型边界上的逻辑形状，不规定底层缓存必须连续、分页或位于哪种设备。
    """

    keys: Tensor
    values: Tensor

    def __post_init__(self) -> None:
        if self.keys.ndim != 4:
            raise ValueError("keys must have shape [batch, sequence, kv_heads, head_size]")
        if self.values.shape != self.keys.shape:
            raise ValueError("keys and values must have the same shape")
        if self.keys.shape[0] <= 0 or self.keys.shape[2] <= 0 or self.keys.shape[3] <= 0:
            raise ValueError("key/value batch, head count and head size must be positive")
        if self.values.dtype != self.keys.dtype or self.values.device != self.keys.device:
            raise ValueError("keys and values must use the same dtype and device")


@dataclass(frozen=True, slots=True)
class KVCacheState:
    """模型一次 forward 读取或新增的逐层 K/V。"""

    layers: tuple[LayerKeyValues, ...]

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        if not layers:
            raise ValueError("KV cache state must contain at least one layer")
        batch_and_sequence = layers[0].keys.shape[:2]
        if any(layer.keys.shape[:2] != batch_and_sequence for layer in layers[1:]):
            raise ValueError("all KV cache layers must share batch and sequence dimensions")
        object.__setattr__(self, "layers", layers)

    @property
    def num_tokens(self) -> int:
        return self.layers[0].keys.shape[1]


@dataclass(frozen=True, slots=True)
class ForwardBatch:
    """传给模型的一批输入 token。

    ``input_ids`` 的形状固定为 ``[batch, padded_sequence]``；
    ``positions`` 使用相同形状，记录每个 token 在请求中的绝对位置；
    ``sequence_lengths`` 记录每行补齐前的有效长度。单请求或等长批次可以
    省略 positions 和长度，此时 positions 从零开始、每一行都使用完整宽度。
    ``kv_cache`` 是这些输入之前已经计算过的历史，不包含本轮 token。
    """

    input_ids: Tensor
    positions: Tensor | None = None
    sequence_lengths: tuple[int, ...] | None = None
    kv_cache: KVCacheState | None = None
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
        if self.kv_cache is not None and self.kv_cache.layers[0].keys.shape[0] != batch_size:
            raise ValueError("KV cache batch size must match input_ids")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "sequence_lengths", lengths)


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """模型一次计算的 logits，以及可选的本轮新增 K/V。"""

    logits: Tensor
    kv_cache_updates: KVCacheState | None = None


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
