"""Qwen2 / Qwen2.5 的可缓存 causal language model。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from light_vllm.modeling.attention.interfaces import (
    AttentionLayerSpec,
    ModelKVCacheSpec,
)
from light_vllm.modeling.models.interfaces import (
    ForwardBatch,
    ModelOutput,
    ModelSpec,
)


@dataclass(frozen=True, slots=True)
class Qwen2Config:
    """当前运行时实际支持的 Qwen2 配置。"""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int = 32_768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False
    pad_token_id: int | None = None

    def __post_init__(self) -> None:
        dimensions = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.max_position_embeddings,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("Qwen2 dimensions must be positive integers")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_size % 2:
            raise ValueError("Qwen2 attention head size must be even")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0 or self.initializer_range <= 0:
            raise ValueError("Qwen2 floating-point configuration values must be positive")
        if self.pad_token_id is not None and (
            type(self.pad_token_id) is not int
            or self.pad_token_id < 0
            or self.pad_token_id >= self.vocab_size
        ):
            raise ValueError("pad_token_id must be within the vocabulary")

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> Qwen2Config:
        """从 HF/ModelScope 的 config.json 读取我们已经支持的字段。"""

        # 不支持的结构直接拒绝，不能悄悄换成相近算法继续计算。
        model_type = values.get("model_type")
        if model_type is not None and model_type != "qwen2":
            raise ValueError(f"expected a qwen2 config, got {model_type!r}")
        if values.get("hidden_act", "silu") != "silu":
            raise ValueError("only Qwen2 silu activation is supported")
        if values.get("use_sliding_window", False):
            raise ValueError("Qwen2 sliding-window attention is not supported yet")
        layer_types = values.get("layer_types")
        if layer_types is not None and any(value != "full_attention" for value in layer_types):
            raise ValueError("only full-attention Qwen2 layers are supported")

        # 新旧版配置存放 RoPE 参数的位置不同，这里归一成一个 rope_theta。
        rope_theta = values.get("rope_theta", 10_000.0)
        rope_parameters = values.get("rope_parameters")
        if rope_parameters is not None:
            if not isinstance(rope_parameters, Mapping):
                raise ValueError("rope_parameters must be an object")
            if rope_parameters.get("rope_type", "default") != "default":
                raise ValueError("only default Qwen2 RoPE is supported")
            rope_theta = rope_parameters.get("rope_theta", rope_theta)
        if values.get("rope_scaling") is not None:
            raise ValueError("Qwen2 rope_scaling is not supported yet")

        required = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
        )
        missing = tuple(name for name in required if name not in values)
        if missing:
            raise ValueError(f"Qwen2 config is missing required fields: {missing!r}")
        num_attention_heads = values["num_attention_heads"]
        num_key_value_heads = values.get("num_key_value_heads", num_attention_heads)
        config = cls(
            vocab_size=values["vocab_size"],
            hidden_size=values["hidden_size"],
            intermediate_size=values["intermediate_size"],
            num_hidden_layers=values["num_hidden_layers"],
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=values.get("max_position_embeddings", 32_768),
            rms_norm_eps=values.get("rms_norm_eps", 1e-6),
            rope_theta=rope_theta,
            initializer_range=values.get("initializer_range", 0.02),
            tie_word_embeddings=values.get("tie_word_embeddings", False),
            pad_token_id=values.get("pad_token_id"),
        )
        configured_head_size = values.get("head_dim")
        if configured_head_size is not None and configured_head_size != config.head_size:
            raise ValueError("custom Qwen2 head_dim is not supported")
        return config


class Qwen2RMSNorm(nn.Module):
    """按每个 token 的均方根缩放隐藏状态。"""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self._eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        variance = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self._eps)
        return self.weight * normalized.to(input_dtype)


class Qwen2MLP(nn.Module):
    """Qwen2 的门控前馈网络。"""

    def __init__(
        self,
        config: Qwen2Config,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            **factory_kwargs,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            **factory_kwargs,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            **factory_kwargs,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def _rotate_half(values: Tensor) -> Tensor:
    """交换并旋转向量的前后两半，供 RoPE 使用。"""

    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    queries: Tensor,
    keys: Tensor,
    cosines: Tensor,
    sines: Tensor,
) -> tuple[Tensor, Tensor]:
    """把绝对位置编码到 query 和 key。"""

    cosines = cosines.unsqueeze(2)
    sines = sines.unsqueeze(2)
    return (
        queries * cosines + _rotate_half(queries) * sines,
        keys * cosines + _rotate_half(keys) * sines,
    )


class Qwen2Attention(nn.Module):
    """生成 Q/K/V，并把实际 attention 交给当前执行后端。"""

    def __init__(
        self,
        config: Qwen2Config,
        layer_index: int,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self._layer_id = f"model.layers.{layer_index}.self_attn"
        self._num_query_heads = config.num_attention_heads
        self._num_kv_heads = config.num_key_value_heads
        self._head_size = config.head_size
        self._scale = 1 / sqrt(config.head_size)
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_size,
            bias=True,
            **factory_kwargs,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_size,
            bias=True,
            **factory_kwargs,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_size,
            bias=True,
            **factory_kwargs,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_size,
            config.hidden_size,
            bias=False,
            **factory_kwargs,
        )

    def forward(
        self,
        hidden_states: Tensor,
        batch: ForwardBatch,
        cosines: Tensor,
        sines: Tensor,
    ) -> Tensor:
        batch_size, query_width, _ = hidden_states.shape
        queries = self.q_proj(hidden_states).view(
            batch_size,
            query_width,
            self._num_query_heads,
            self._head_size,
        )
        keys = self.k_proj(hidden_states).view(
            batch_size,
            query_width,
            self._num_kv_heads,
            self._head_size,
        )
        values = self.v_proj(hidden_states).view_as(keys)
        queries, keys = _apply_rotary(queries, keys, cosines, sines)

        attention = batch.attention
        if attention is None:
            raise ValueError("Qwen2 requires an attention context")
        # 模型只描述 Q/K/V；缓存布局、softmax 和 kernel 由执行端选择。
        attended = attention.forward(
            self._layer_id,
            queries,
            keys,
            values,
            scale=self._scale,
        )
        output = self.o_proj(attended.reshape(batch_size, query_width, -1))
        return output


class Qwen2DecoderLayer(nn.Module):
    """一层 self-attention、前馈网络和两次残差连接。"""

    def __init__(
        self,
        config: Qwen2Config,
        layer_index: int,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = Qwen2Attention(config, layer_index, **factory_kwargs)
        self.mlp = Qwen2MLP(config, **factory_kwargs)
        self.input_layernorm = Qwen2RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            **factory_kwargs,
        )
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            **factory_kwargs,
        )

    def forward(
        self,
        hidden_states: Tensor,
        batch: ForwardBatch,
        cosines: Tensor,
        sines: Tensor,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            batch,
            cosines,
            sines,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class Qwen2Model(nn.Module):
    """词向量、连续多层 Decoder 和最终归一化。"""

    def __init__(
        self,
        config: Qwen2Config,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            **factory_kwargs,
        )
        self.layers = nn.ModuleList(
            Qwen2DecoderLayer(config, layer_index, **factory_kwargs)
            for layer_index in range(config.num_hidden_layers)
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps, **factory_kwargs)


class Qwen2ForCausalLM(nn.Module):
    """只实现推理所需的 Qwen2 路径，权重名称与 HF checkpoint 对齐。"""

    def __init__(
        self,
        config: Qwen2Config,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        self.model = Qwen2Model(config, **factory_kwargs)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            **factory_kwargs,
        )
        self.apply(self._initialize_weights)
        if config.tie_word_embeddings:
            # 输入词向量和输出分类层可以共用同一份参数。
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def from_spec(cls, spec: ModelSpec) -> Qwen2ForCausalLM:
        return cls(
            Qwen2Config.from_mapping(spec.model_args),
            device=spec.device,
            dtype=spec.dtype,
        )

    @property
    def max_model_tokens(self) -> int:
        return self.config.max_position_embeddings

    @property
    def kv_cache_spec(self) -> ModelKVCacheSpec:
        """告诉执行端每一层需要怎样的 K/V 张量。"""

        return ModelKVCacheSpec(
            layers=tuple(
                AttentionLayerSpec(
                    layer_id=f"model.layers.{layer_index}.self_attn",
                    num_query_heads=self.config.num_attention_heads,
                    num_kv_heads=self.config.num_key_value_heads,
                    head_size=self.config.head_size,
                )
                for layer_index in range(self.config.num_hidden_layers)
            )
        )

    @property
    def optional_weight_keys(self) -> frozenset[str]:
        """共享 lm_head 时，快照只保存词向量也能完整加载。"""

        if self.config.tie_word_embeddings:
            return frozenset({"lm_head.weight"})
        return frozenset()

    def forward(self, batch: ForwardBatch) -> ModelOutput:
        """执行所有 Decoder 层；attention 计算统一交给当前上下文。"""

        if batch.attention is None:
            raise ValueError("Qwen2 requires an attention context")

        positions = batch.positions
        assert positions is not None
        if bool(torch.any(positions >= self.config.max_position_embeddings)):
            raise ValueError("Qwen2 position exceeds max_position_embeddings")
        hidden_states = self.model.embed_tokens(batch.input_ids)
        cosines, sines = self._rotary_embeddings(positions, hidden_states.dtype)
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states,
                batch,
                cosines,
                sines,
            )
        logits = self.lm_head(self.model.norm(hidden_states))
        return ModelOutput(logits=logits)

    def _rotary_embeddings(
        self,
        positions: Tensor,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """按绝对位置生成 RoPE 使用的正弦和余弦。"""

        frequencies = 1.0 / (
            self.config.rope_theta
            ** (
                torch.arange(
                    0,
                    self.config.head_size,
                    2,
                    dtype=torch.float32,
                    device=positions.device,
                )
                / self.config.head_size
            )
        )
        angles = torch.einsum("bs,d->bsd", positions.float(), frequencies)
        embeddings = torch.cat((angles, angles), dim=-1)
        return embeddings.cos().to(dtype), embeddings.sin().to(dtype)

    def _initialize_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
