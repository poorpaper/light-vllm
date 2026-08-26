from __future__ import annotations

from light_vllm.modeling.catalog import Catalog
from light_vllm.modeling.loaders.safetensors import SafetensorsModelLoader
from light_vllm.modeling.loaders.torch import InitModelLoader, StateDictModelLoader
from light_vllm.modeling.models.qwen2 import Qwen2ForCausalLM
from light_vllm.modeling.models.tiny import TinyCausalLM
from light_vllm.modeling.models.tiny_attention import TinyAttentionCausalLM
from light_vllm.modeling.quantization.awq import create_awq_linear_method
from light_vllm.modeling.runner import ModelRunner


def create_catalog() -> Catalog:
    catalog = Catalog()
    catalog.models.register("tiny-causal-lm", TinyCausalLM.from_spec)
    catalog.models.register("tiny-attention-causal-lm", TinyAttentionCausalLM.from_spec)
    catalog.models.register("qwen2", Qwen2ForCausalLM.from_spec)
    # HF 的 Qwen2.5 checkpoint 仍声明 model_type=qwen2，并复用相同权重结构。
    # 单独注册用户可见的名字即可，不需要在 runner 或模型里增加版本分支。
    catalog.models.register("qwen2.5", Qwen2ForCausalLM.from_spec)
    catalog.quantizations.register("awq", create_awq_linear_method)
    catalog.loaders.register("init", InitModelLoader())
    catalog.loaders.register("state-dict", StateDictModelLoader())
    catalog.loaders.register(
        "safetensors", SafetensorsModelLoader(quantizations=catalog.quantizations)
    )
    return catalog


def create_runner(catalog: Catalog | None = None) -> ModelRunner:
    return ModelRunner(catalog or create_catalog())
