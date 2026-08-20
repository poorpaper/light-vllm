"""带单层 causal self-attention 的最小可缓存模型。"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn

from light_vllm.modeling.attention.interfaces import AttentionLayerSpec, ModelKVCacheSpec
from light_vllm.modeling.models.interfaces import (
    ForwardBatch,
    ModelOutput,
    ModelSpec,
    select_query_states,
)


@dataclass(frozen=True, slots=True)
class TinyAttentionConfig:
    vocab_size: int = 256
    hidden_size: int = 64
    num_heads: int = 4

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value <= 0
            for value in (self.vocab_size, self.hidden_size, self.num_heads)
        ):
            raise ValueError("model dimensions must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")


class TinyAttentionCausalLM(nn.Module):
    """用于验证全序列计算与 KV cache 计算等价的最小 attention 模型。"""

    def __init__(self, config: TinyAttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_spec(cls, spec: ModelSpec) -> TinyAttentionCausalLM:
        return cls(TinyAttentionConfig(**spec.model_args))

    @property
    def head_size(self) -> int:
        return self.config.hidden_size // self.config.num_heads

    @property
    def kv_cache_spec(self) -> ModelKVCacheSpec:
        return ModelKVCacheSpec(
            layers=(
                AttentionLayerSpec(
                    layer_id="attention",
                    num_query_heads=self.config.num_heads,
                    num_kv_heads=self.config.num_heads,
                    head_size=self.head_size,
                ),
            )
        )

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        hidden_states = self.token_embedding(batch.input_ids)
        queries = self._split_heads(self.query(hidden_states))
        new_keys = self._split_heads(self.key(hidden_states))
        new_values = self._split_heads(self.value(hidden_states))

        if batch.attention is None:
            raise ValueError("TinyAttentionCausalLM requires an attention context")
        attended = batch.attention.forward(
            "attention",
            queries,
            new_keys,
            new_values,
            scale=1 / sqrt(self.head_size),
        )

        attended = attended.reshape(*hidden_states.shape)
        hidden_states = select_query_states(self.output(attended), batch)
        logits = self.lm_head(hidden_states)
        return ModelOutput(logits=logits)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(
            tensor.shape[0],
            tensor.shape[1],
            self.config.num_heads,
            self.head_size,
        )
