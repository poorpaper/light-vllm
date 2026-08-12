"""模型 logits 到 token ID 的采样策略。"""

from __future__ import annotations

from typing import Protocol

from torch import Tensor


class SamplingError(RuntimeError):
    """采样输入或输出违反契约时抛出。"""


class Sampler(Protocol):
    """从每个 batch row 的词表 logits 中选择一个 token。"""

    def sample(self, logits: Tensor) -> tuple[int, ...]: ...


class GreedySampler:
    """为每一行选择分数最高的 token。"""

    def sample(self, logits: Tensor) -> tuple[int, ...]:
        if logits.ndim != 2:
            raise SamplingError("sampler logits must have shape [batch, vocabulary]")
        if logits.shape[0] == 0 or logits.shape[1] == 0:
            raise SamplingError("sampler logits must not be empty")
        return tuple(int(token_id) for token_id in logits.argmax(dim=-1).tolist())
