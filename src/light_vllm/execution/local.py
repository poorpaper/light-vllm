from __future__ import annotations

import torch

from light_vllm.execution.api import ExecutionError, ExecutionNotReadyError
from light_vllm.models.api import ForwardBatch, ModelForwarder, ModelNotLoadedError


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
        """准备模型输入，并返回本轮选中的 token。"""

        if not token_ids:
            raise ExecutionError("token_ids must not be empty")

        batch = ForwardBatch(
            input_ids=torch.tensor([token_ids], dtype=torch.long, device=self._device)
        )
        try:
            output = self._runner.forward(batch)
        except ModelNotLoadedError as exc:
            raise ExecutionNotReadyError("load a model before executing") from exc

        if output.logits.ndim != 3 or output.logits.shape[:2] != batch.input_ids.shape:
            raise ExecutionError("model logits must have shape [batch, sequence, vocabulary]")

        return int(output.logits[0, -1].argmax().item())
