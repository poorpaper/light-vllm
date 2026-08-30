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
from light_vllm.modeling.models._qwen2_kernels import (
    apply_rotary,
    fused_add_rms_norm,
    silu_and_mul,
)
from light_vllm.modeling.models.interfaces import (
    ForwardBatch,
    ModelOutput,
    ModelSpec,
    select_query_states,
)
from light_vllm.modeling.quantization.dense import DenseLinearMethod
from light_vllm.modeling.quantization.interfaces import (
    DirectLinear,
    LinearMethod,
    LinearOperation,
    RowParallelLayer,
)
from light_vllm.modeling.tensor_parallel import (
    TensorParallelContext,
    VocabParallelEmbedding,
    VocabParallelLinear,
    partition_dimension,
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
        return F.rms_norm(
            hidden_states,
            (hidden_states.shape[-1],),
            self.weight,
            self._eps,
        )

    @property
    def eps(self) -> float:
        return self._eps


class Qwen2MLP(nn.Module):
    """Qwen2 的门控前馈网络。"""

    def __init__(
        self,
        config: Qwen2Config,
        layer_index: int,
        parallel: TensorParallelContext,
        linear_method: LinearMethod,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self._linear_method = linear_method
        self.gate_proj = linear_method.create_column(
            config.hidden_size,
            config.intermediate_size,
            parallel,
            prefix=f"model.layers.{layer_index}.mlp.gate_proj",
            bias=False,
            **factory_kwargs,
        )
        self.up_proj = linear_method.create_column(
            config.hidden_size,
            config.intermediate_size,
            parallel,
            prefix=f"model.layers.{layer_index}.mlp.up_proj",
            bias=False,
            **factory_kwargs,
        )
        self.down_proj = linear_method.create_row(
            config.intermediate_size,
            config.hidden_size,
            parallel,
            prefix=f"model.layers.{layer_index}.mlp.down_proj",
            bias=False,
            **factory_kwargs,
        )
        if not isinstance(self.down_proj, RowParallelLayer):
            raise TypeError("LinearMethod.create_row must return a row-parallel layer")
        self._needs_output_reduction = parallel.world_size > 1
        self.register_buffer("_merged_gate_up_weight_t", None, persistent=False)
        self.register_buffer("_down_proj_weight_t", None, persistent=False)
        self._merged_gate_up_operation: LinearOperation | None = None
        self._down_proj_operation: LinearOperation | None = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        merged_weight_t = self._merged_gate_up_weight_t
        down_weight_t = self._down_proj_weight_t
        if merged_weight_t is not None:
            gate, up = torch.mm(hidden_states, merged_weight_t).chunk(2, dim=-1)
        elif self._merged_gate_up_operation is not None:
            gate, up = self._merged_gate_up_operation(hidden_states).chunk(2, dim=-1)
        else:
            gate = self.gate_proj(hidden_states)
            up = self.up_proj(hidden_states)
        activated = silu_and_mul(gate, up)
        if down_weight_t is not None:
            local_output = torch.mm(activated, down_weight_t)
        elif self._down_proj_operation is not None:
            local_output = self._down_proj_operation(activated)
        else:
            return self.down_proj(activated)
        if not self._needs_output_reduction:
            # TP=1 保留原来的直通热路径，不为无通信场景调用 collective 适配层。
            return local_output
        return self.down_proj.reduce_output(local_output)

    def prepare_for_inference(self) -> None:
        """Pack checkpoint-compatible Gate/Up weights into one inference GEMM."""

        merged = self._linear_method.prepare_merged((self.gate_proj, self.up_proj))
        down = self._linear_method.prepare_local(self.down_proj)
        if isinstance(merged, DirectLinear):
            if merged.bias is not None:
                raise ValueError("Qwen2 gate/up projections must not have bias")
            self._merged_gate_up_weight_t = merged.weight_t
        else:
            self._merged_gate_up_operation = merged
        if isinstance(down, DirectLinear):
            if down.bias is not None:
                raise ValueError("Qwen2 down projection must not have bias")
            self._down_proj_weight_t = down.weight_t
        else:
            self._down_proj_operation = down


class Qwen2Attention(nn.Module):
    """生成 Q/K/V，并把实际 attention 交给当前执行后端。"""

    def __init__(
        self,
        config: Qwen2Config,
        layer_index: int,
        parallel: TensorParallelContext,
        linear_method: LinearMethod,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self._layer_id = f"model.layers.{layer_index}.self_attn"
        tp_size = parallel.world_size
        if config.num_attention_heads % tp_size:
            raise ValueError("Qwen2 query heads must be divisible by tensor parallel size")
        if config.num_key_value_heads >= tp_size:
            if config.num_key_value_heads % tp_size:
                raise ValueError("Qwen2 KV heads must be divisible by tensor parallel size")
            kv_rank = parallel.rank
            kv_world_size = tp_size
        else:
            if tp_size % config.num_key_value_heads:
                raise ValueError(
                    "tensor parallel size must be divisible by Qwen2 KV heads when replicating"
                )
            # KV head 少于 Rank 时，让相邻 Rank 共享同一个 KV head。每个 Rank
            # 仍有独立物理 KV，不需要改变 attention backend。
            replicas_per_kv_head = tp_size // config.num_key_value_heads
            kv_rank = parallel.rank // replicas_per_kv_head
            kv_world_size = config.num_key_value_heads
        self._num_query_heads = config.num_attention_heads // tp_size
        self._num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self._head_size = config.head_size
        self._scale = 1 / sqrt(config.head_size)
        kv_output_partition = partition_dimension(
            config.num_key_value_heads * config.head_size,
            kv_rank,
            kv_world_size,
        )
        self._linear_method = linear_method
        self.q_proj = linear_method.create_column(
            config.hidden_size,
            config.num_attention_heads * config.head_size,
            parallel,
            prefix=f"model.layers.{layer_index}.self_attn.q_proj",
            bias=True,
            **factory_kwargs,
        )
        self.k_proj = linear_method.create_column(
            config.hidden_size,
            config.num_key_value_heads * config.head_size,
            parallel,
            prefix=f"model.layers.{layer_index}.self_attn.k_proj",
            bias=True,
            output_partition=kv_output_partition,
            **factory_kwargs,
        )
        self.v_proj = linear_method.create_column(
            config.hidden_size,
            config.num_key_value_heads * config.head_size,
            parallel,
            prefix=f"model.layers.{layer_index}.self_attn.v_proj",
            bias=True,
            output_partition=kv_output_partition,
            **factory_kwargs,
        )
        self.o_proj = linear_method.create_row(
            config.num_attention_heads * config.head_size,
            config.hidden_size,
            parallel,
            prefix=f"model.layers.{layer_index}.self_attn.o_proj",
            bias=False,
            **factory_kwargs,
        )
        if not isinstance(self.o_proj, RowParallelLayer):
            raise TypeError("LinearMethod.create_row must return a row-parallel layer")
        self._needs_output_reduction = parallel.world_size > 1
        self.register_buffer("_merged_qkv_weight_t", None, persistent=False)
        self.register_buffer("_merged_qkv_bias", None, persistent=False)
        self.register_buffer("_o_proj_weight_t", None, persistent=False)
        self._merged_qkv_operation: LinearOperation | None = None
        self._o_proj_operation: LinearOperation | None = None

    def forward(
        self,
        hidden_states: Tensor,
        batch: ForwardBatch,
        cosines: Tensor,
        sines: Tensor,
    ) -> Tensor:
        num_tokens = hidden_states.shape[0]
        merged_weight_t = self._merged_qkv_weight_t
        merged_bias = self._merged_qkv_bias
        output_weight_t = self._o_proj_weight_t
        if merged_weight_t is not None and merged_bias is not None:
            projections = torch.addmm(merged_bias, hidden_states, merged_weight_t)
            query_size = self._num_query_heads * self._head_size
            kv_size = self._num_kv_heads * self._head_size
            query_projection, key_projection, value_projection = projections.split(
                (query_size, kv_size, kv_size),
                dim=-1,
            )
        elif self._merged_qkv_operation is not None:
            projections = self._merged_qkv_operation(hidden_states)
            query_size = self._num_query_heads * self._head_size
            kv_size = self._num_kv_heads * self._head_size
            query_projection, key_projection, value_projection = projections.split(
                (query_size, kv_size, kv_size),
                dim=-1,
            )
        else:
            query_projection = self.q_proj(hidden_states)
            key_projection = self.k_proj(hidden_states)
            value_projection = self.v_proj(hidden_states)
        queries = query_projection.view(
            num_tokens,
            self._num_query_heads,
            self._head_size,
        )
        keys = key_projection.view(
            num_tokens,
            self._num_kv_heads,
            self._head_size,
        )
        values = value_projection.view_as(keys)
        queries, keys = apply_rotary(queries, keys, cosines, sines)

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
        attended = attended.reshape(num_tokens, -1)
        if output_weight_t is not None:
            local_output = torch.mm(attended, output_weight_t)
        elif self._o_proj_operation is not None:
            local_output = self._o_proj_operation(attended)
        else:
            return self.o_proj(attended)
        if not self._needs_output_reduction:
            return local_output
        return self.o_proj.reduce_output(local_output)

    def prepare_for_inference(self) -> None:
        """Pack checkpoint-compatible Q/K/V weights into one inference GEMM."""

        merged = self._linear_method.prepare_merged((self.q_proj, self.k_proj, self.v_proj))
        output = self._linear_method.prepare_local(self.o_proj)
        if isinstance(merged, DirectLinear):
            if merged.bias is None:
                raise ValueError("Qwen2 Q/K/V projections require bias")
            self._merged_qkv_weight_t = merged.weight_t
            self._merged_qkv_bias = merged.bias
        else:
            self._merged_qkv_operation = merged
        if isinstance(output, DirectLinear):
            if output.bias is not None:
                raise ValueError("Qwen2 output projection must not have bias")
            self._o_proj_weight_t = output.weight_t
        else:
            self._o_proj_operation = output


class Qwen2DecoderLayer(nn.Module):
    """一层 self-attention、前馈网络和两次残差连接。"""

    def __init__(
        self,
        config: Qwen2Config,
        layer_index: int,
        parallel: TensorParallelContext,
        linear_method: LinearMethod,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = Qwen2Attention(
            config, layer_index, parallel, linear_method, **factory_kwargs
        )
        self.mlp = Qwen2MLP(config, layer_index, parallel, linear_method, **factory_kwargs)
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
        residual: Tensor | None,
        batch: ForwardBatch,
        cosines: Tensor,
        sines: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = fused_add_rms_norm(
                hidden_states,
                residual,
                self.input_layernorm.weight,
                self.input_layernorm.eps,
            )
        hidden_states = self.self_attn(
            hidden_states,
            batch,
            cosines,
            sines,
        )
        hidden_states, residual = fused_add_rms_norm(
            hidden_states,
            residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        return self.mlp(hidden_states), residual

    def prepare_for_inference(self) -> None:
        self.self_attn.prepare_for_inference()
        self.mlp.prepare_for_inference()


class Qwen2Model(nn.Module):
    """词向量、连续多层 Decoder 和最终归一化。"""

    def __init__(
        self,
        config: Qwen2Config,
        parallel: TensorParallelContext,
        linear_method: LinearMethod,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            parallel,
            config.pad_token_id,
            **factory_kwargs,
        )
        self.layers = nn.ModuleList(
            Qwen2DecoderLayer(config, layer_index, parallel, linear_method, **factory_kwargs)
            for layer_index in range(config.num_hidden_layers)
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps, **factory_kwargs)


class Qwen2ForCausalLM(nn.Module):
    """只实现推理所需的 Qwen2 路径，权重名称与 HF checkpoint 对齐。"""

    def __init__(
        self,
        config: Qwen2Config,
        parallel: TensorParallelContext | None = None,
        linear_method: LinearMethod | None = None,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        parallel = parallel or TensorParallelContext()
        linear_method = linear_method or DenseLinearMethod()
        if config.intermediate_size % parallel.world_size:
            raise ValueError("Qwen2 intermediate size must be divisible by tensor parallel size")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        self._parallel = parallel
        self._needs_logits_gather = parallel.world_size > 1
        self.model = Qwen2Model(config, parallel, linear_method, **factory_kwargs)
        self.lm_head = VocabParallelLinear(
            config.hidden_size,
            config.vocab_size,
            parallel,
            bias=False,
            **factory_kwargs,
        )
        self.register_buffer("_rope_cosines", None, persistent=False)
        self.register_buffer("_rope_sines", None, persistent=False)
        self.register_buffer("_lm_head_weight_t", None, persistent=False)
        self.apply(self._initialize_weights)
        if config.tie_word_embeddings:
            # 输入词向量和输出分类层可以共用同一份参数。
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def from_spec(cls, spec: ModelSpec) -> Qwen2ForCausalLM:
        return cls(
            Qwen2Config.from_mapping(spec.model_args),
            parallel=spec.tensor_parallel,
            linear_method=spec.linear_method,
            device=spec.device,
            dtype=spec.dtype,
        )

    @property
    def max_model_tokens(self) -> int:
        return self.config.max_position_embeddings

    @property
    def tensor_parallel_size(self) -> int:
        return self._parallel.world_size

    @property
    def kv_cache_spec(self) -> ModelKVCacheSpec:
        """告诉执行端每一层需要怎样的 K/V 张量。"""

        return ModelKVCacheSpec(
            layers=tuple(
                AttentionLayerSpec(
                    layer_id=f"model.layers.{layer_index}.self_attn",
                    num_query_heads=layer.self_attn._num_query_heads,
                    num_kv_heads=layer.self_attn._num_kv_heads,
                    head_size=self.config.head_size,
                )
                for layer_index, layer in enumerate(self.model.layers)
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
        if not batch.positions_are_validated:
            positions_in_range = torch.all(positions < self.config.max_position_embeddings)
            if positions.device.type == "cuda":
                torch._assert_async(
                    positions_in_range,
                    "Qwen2 position exceeds max_position_embeddings",
                )
            elif not bool(positions_in_range):
                raise ValueError("Qwen2 position exceeds max_position_embeddings")
        hidden_states = self.model.embed_tokens(batch.input_ids)
        cosines, sines = self._rotary_embeddings(positions, hidden_states.dtype)
        residual = None
        for layer in self.model.layers:
            hidden_states, residual = layer(
                hidden_states,
                residual,
                batch,
                cosines,
                sines,
            )
        assert residual is not None
        hidden_states, _ = fused_add_rms_norm(
            hidden_states,
            residual,
            self.model.norm.weight,
            self.model.norm.eps,
        )
        hidden_states = select_query_states(hidden_states, batch)
        if self._lm_head_weight_t is None:
            logits = self.lm_head(hidden_states)
        else:
            local_logits = torch.mm(hidden_states, self._lm_head_weight_t)
            logits = (
                self.lm_head.gather_output(local_logits)
                if self._needs_logits_gather
                else local_logits
            )
        return ModelOutput(logits=logits)

    def prepare_for_inference(self) -> None:
        """Build model-owned packed weights after checkpoint loading completes."""

        for layer in self.model.layers:
            layer.prepare_for_inference()
        # lm_head 与 embedding 可能共享参数；转置 view 不复制这份大权重。
        self._lm_head_weight_t = self.lm_head.weight.t()
        positions = torch.arange(
            self.config.max_position_embeddings,
            dtype=torch.float32,
            device=self.model.embed_tokens.weight.device,
        )
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
        angles = torch.outer(positions, frequencies)
        embeddings = torch.cat((angles, angles), dim=-1)
        dtype = self.model.embed_tokens.weight.dtype
        self._rope_cosines = embeddings.cos().to(dtype)
        self._rope_sines = embeddings.sin().to(dtype)

    def _rotary_embeddings(
        self,
        positions: Tensor,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """按绝对位置生成 RoPE 使用的正弦和余弦。"""

        if self._rope_cosines is not None and self._rope_sines is not None:
            return self._rope_cosines[positions], self._rope_sines[positions]

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
        angles = torch.einsum("s,d->sd", positions.float(), frequencies)
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
