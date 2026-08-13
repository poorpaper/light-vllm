"""带单层 causal self-attention 的最小可缓存模型。"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn

from light_vllm.modeling.models.interfaces import (
    ForwardBatch,
    KVCacheState,
    LayerKeyValues,
    ModelOutput,
    ModelSpec,
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

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        hidden_states = self.token_embedding(batch.input_ids)
        queries = self._split_heads(self.query(hidden_states))
        new_keys = self._split_heads(self.key(hidden_states))
        new_values = self._split_heads(self.value(hidden_states))

        past_length = 0
        keys = new_keys
        values = new_values
        if batch.kv_cache is not None:
            if len(batch.kv_cache.layers) != 1:
                raise ValueError("TinyAttentionCausalLM expects one KV cache layer")
            past = batch.kv_cache.layers[0]
            self._validate_past(past, batch.input_ids.shape[0])
            past_length = past.keys.shape[1]
            keys = torch.cat((past.keys, new_keys), dim=1)
            values = torch.cat((past.values, new_values), dim=1)

        # 第 i 个新 query 可以读取全部历史，以及本轮不晚于 i 的 key。
        query_positions = past_length + torch.arange(queries.shape[1], device=queries.device)
        key_positions = torch.arange(keys.shape[1], device=keys.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

        scores = torch.einsum("bshd,bthd->bhst", queries, keys) / sqrt(self.head_size)
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.einsum("bhst,bthd->bshd", probabilities, values)
        attended = attended.reshape(*hidden_states.shape)
        logits = self.lm_head(self.output(attended))

        updates = None
        if batch.kv_cache is not None:
            updates = KVCacheState(layers=(LayerKeyValues(new_keys, new_values),))
        return ModelOutput(logits=logits, kv_cache_updates=updates)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(
            tensor.shape[0],
            tensor.shape[1],
            self.config.num_heads,
            self.head_size,
        )

    def _validate_past(self, past: LayerKeyValues, batch_size: int) -> None:
        expected_tail = (self.config.num_heads, self.head_size)
        if past.keys.shape[0] != batch_size or past.keys.shape[2:] != expected_tail:
            raise ValueError("KV cache shape does not match TinyAttentionCausalLM")
        parameter = self.token_embedding.weight
        if past.keys.dtype != parameter.dtype or past.keys.device != parameter.device:
            raise ValueError("KV cache dtype and device must match TinyAttentionCausalLM")
