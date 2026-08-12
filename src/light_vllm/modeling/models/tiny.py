from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from light_vllm.modeling.models.interfaces import ForwardBatch, ModelOutput, ModelSpec


@dataclass(frozen=True, slots=True)
class TinyCausalLMConfig:
    vocab_size: int = 256
    hidden_size: int = 64


class TinyCausalLM(nn.Module):
    """用于测试整体流程的最小模型。"""

    def __init__(self, config: TinyCausalLMConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_spec(cls, spec: ModelSpec) -> TinyCausalLM:
        return cls(TinyCausalLMConfig(**spec.model_args))

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        hidden_states = self.token_embedding(batch.input_ids)
        return ModelOutput(logits=self.lm_head(hidden_states))
