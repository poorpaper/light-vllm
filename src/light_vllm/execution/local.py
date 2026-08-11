"""本地 PyTorch 贪心执行器。

单请求与批量执行共用模型错误映射和 logits 形状检查，但保持两个独立契约：
reference 路径不会因为 batching 需求而变复杂，批量路径也不需要理解生成事件。
"""

from __future__ import annotations

import torch

from light_vllm.execution.api import (
    ExecutionBatch,
    ExecutionError,
    ExecutionNotReadyError,
    TokenSelection,
)
from light_vllm.models.api import ForwardBatch, ModelForwarder, ModelNotLoadedError, ModelOutput


def _forward(runner: ModelForwarder, batch: ForwardBatch) -> ModelOutput:
    """执行统一模型边界，并把模型生命周期错误转换成执行层错误。"""

    try:
        output = runner.forward(batch)
    except ModelNotLoadedError as exc:
        raise ExecutionNotReadyError("load a model before executing") from exc

    if output.logits.ndim != 3 or output.logits.shape[:2] != batch.input_ids.shape:
        raise ExecutionError("model logits must have shape [batch, sequence, vocabulary]")
    return output


class GreedyTokenExecutor:
    """用本地 PyTorch 模型计算并选择分数最高的 token。"""

    def __init__(
        self,
        runner: ModelForwarder,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self._runner = runner
        self._device = torch.device(device)

    @property
    def ready(self) -> bool:
        return self._runner.generation > 0

    def next_token(self, token_ids: tuple[int, ...]) -> int:
        """重算一个请求的完整序列，并返回本轮选中的 token。"""

        if not token_ids:
            raise ExecutionError("token_ids must not be empty")

        batch = ForwardBatch(
            input_ids=torch.tensor([token_ids], dtype=torch.long, device=self._device)
        )
        output = _forward(self._runner, batch)

        return int(output.logits[0, -1].argmax().item())


class GreedyBatchTokenExecutor:
    """右侧补齐不同长度的序列，并批量选择贪心 token。

    当前实现每轮重算完整序列，不管理 KV cache。右侧补齐保证 causal 模型
    的有效位置不会看到后面的 padding；``sequence_lengths`` 仍会显式传给
    模型边界，供后续 attention mask 或其他后端使用。
    """

    def __init__(
        self,
        runner: ModelForwarder,
        *,
        device: str | torch.device = "cpu",
        padding_token_id: int = 0,
    ) -> None:
        if type(padding_token_id) is not int or padding_token_id < 0:
            raise ValueError("padding_token_id must be a non-negative integer")
        self._runner = runner
        self._device = torch.device(device)
        self._padding_token_id = padding_token_id

    @property
    def ready(self) -> bool:
        return self._runner.generation > 0

    def next_tokens(self, batch: ExecutionBatch) -> tuple[TokenSelection, ...]:
        """执行一个右侧补齐的批次，并读取每行最后一个有效位置。"""

        lengths = tuple(len(sequence.token_ids) for sequence in batch.sequences)
        # 先创建统一宽度的矩阵，再逐行覆盖有效 token；padding 只出现在右侧。
        input_ids = torch.full(
            (len(batch.sequences), max(lengths)),
            self._padding_token_id,
            dtype=torch.long,
            device=self._device,
        )
        for row, sequence in enumerate(batch.sequences):
            input_ids[row, : lengths[row]] = torch.tensor(
                sequence.token_ids,
                dtype=torch.long,
                device=self._device,
            )

        forward_batch = ForwardBatch(input_ids=input_ids, sequence_lengths=lengths)
        output = _forward(self._runner, forward_batch)
        # 每行的最后一个有效位置不同，不能直接统一读取 logits[:, -1]。
        return tuple(
            TokenSelection(
                request_id=sequence.request_id,
                token_id=int(output.logits[row, lengths[row] - 1].argmax().item()),
            )
            for row, sequence in enumerate(batch.sequences)
        )
