from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import torch
from torch import Tensor, nn

from light_vllm.modeling.attention.interfaces import AttentionContext, ModelKVCacheSpec
from light_vllm.modeling.tensor_parallel import TensorParallelContext

if TYPE_CHECKING:
    from light_vllm.modeling.quantization.interfaces import LinearMethod


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """加载模型时需要的配置。"""

    architecture: str
    loader: str = "init"
    model_args: Mapping[str, object] = field(default_factory=dict)
    weights: Path | None = None
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32
    tensor_parallel: TensorParallelContext | None = None
    # 这两个字段只描述模型加载策略。loader 解析 checkpoint 元数据后把已经
    # 选好的 LinearMethod 交给模型 factory；Runner 和执行热路径无需理解量化名。
    quantization: str = "auto"
    quantization_backend: str = "auto"
    linear_method: LinearMethod | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class ForwardBatch:
    """传给模型的一批输入 token。

    ``input_ids`` 和 ``positions`` 都是一维 token 流；``query_start_loc``
    用首尾边界把这条流切回各请求。单请求可以省略边界和 positions。
    可缓存 attention 模型必须调用 ``attention``；具体连续或分页布局由执行端注入。
    不含 attention 的模型可以省略这个字段。
    """

    input_ids: Tensor
    positions: Tensor | None = None
    query_start_loc: tuple[int, ...] | None = None
    attention: AttentionContext | None = None
    # None 表示消费全部 token；否则只投影一维 token 流中的指定行。
    logit_query_indices: tuple[int, ...] | None = None
    # Step Handler 已在 CPU 侧验证绝对位置时，模型不再启动重复的 CUDA 检查。
    positions_are_validated: bool = False

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 1 or self.input_ids.numel() == 0:
            raise ValueError("input_ids must be a non-empty one-dimensional token stream")

        num_tokens = self.input_ids.shape[0]
        query_start_loc = self.query_start_loc
        query_start_loc = (0, num_tokens) if query_start_loc is None else tuple(query_start_loc)
        if (
            len(query_start_loc) < 2
            or query_start_loc[0] != 0
            or query_start_loc[-1] != num_tokens
            or any(type(index) is not int for index in query_start_loc)
            or any(
                left >= right
                for left, right in zip(query_start_loc, query_start_loc[1:], strict=False)
            )
        ):
            raise ValueError(
                "query_start_loc must start at zero, end at the token count, and increase"
            )

        positions = self.positions
        if positions is None:
            if len(query_start_loc) != 2:
                raise ValueError("positions are required for packed multi-request batches")
            positions = torch.arange(
                num_tokens,
                dtype=torch.long,
                device=self.input_ids.device,
            )
        if positions.shape != self.input_ids.shape:
            raise ValueError("positions must have the same shape as input_ids")
        if positions.dtype != torch.long or positions.device != self.input_ids.device:
            raise ValueError("positions must use torch.long on the input_ids device")
        if type(self.positions_are_validated) is not bool:
            raise TypeError("positions_are_validated must be a boolean")
        if not self.positions_are_validated:
            positions_non_negative = torch.all(positions >= 0)
            if positions.device.type == "cuda":
                # 契约校验留在当前 CUDA stream，不在模型提交前强制 CPU 同步。
                torch._assert_async(positions_non_negative, "positions must not be negative")
            elif not bool(positions_non_negative):
                raise ValueError("positions must not be negative")
        logit_query_indices = self.logit_query_indices
        if logit_query_indices is not None:
            logit_query_indices = tuple(logit_query_indices)
            if any(
                type(index) is not int or not 0 <= index < num_tokens
                for index in logit_query_indices
            ):
                raise ValueError("logit query indices must select valid token rows")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "query_start_loc", query_start_loc)
        object.__setattr__(self, "logit_query_indices", logit_query_indices)

    @property
    def batch_size(self) -> int:
        assert self.query_start_loc is not None
        return len(self.query_start_loc) - 1

    @property
    def query_lengths(self) -> tuple[int, ...]:
        assert self.query_start_loc is not None
        return tuple(
            end - start
            for start, end in zip(self.query_start_loc, self.query_start_loc[1:], strict=False)
        )

    @property
    def logits_width(self) -> int:
        if self.logit_query_indices is None:
            return self.input_ids.shape[0]
        return len(self.logit_query_indices)


def select_query_states(hidden_states: Tensor, batch: ForwardBatch) -> Tensor:
    """Gather only hidden-state rows whose logits the caller requested."""

    indices = batch.logit_query_indices
    if indices is None:
        return hidden_states
    if not indices:
        return hidden_states[:0]
    if len(indices) == hidden_states.shape[0] and all(
        index == row for row, index in enumerate(indices)
    ):
        # 普通 decode 请求每行都需要 logits；直接复用 hidden states，避免一次
        # 小 tensor H2D 和无意义的 index_select。
        return hidden_states
    gather_indices = torch.tensor(indices, dtype=torch.long, device=hidden_states.device)
    return hidden_states.index_select(0, gather_indices)


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """模型一次计算得到的 requested-query logits。"""

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

    @property
    def tensor_parallel_size(self) -> int: ...

    def forward(self, batch: ForwardBatch) -> ModelOutput: ...


class ModelSessionProvider(Protocol):
    """让执行代码取得当前模型，并在本次执行期间继续使用它。"""

    @property
    def generation(self) -> int: ...

    def open_session(self) -> ModelSession: ...
