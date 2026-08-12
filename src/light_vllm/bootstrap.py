from __future__ import annotations

from light_vllm.modeling.catalog import Catalog
from light_vllm.modeling.loaders.torch import InitModelLoader, StateDictModelLoader
from light_vllm.modeling.models.tiny import TinyCausalLM
from light_vllm.modeling.models.tiny_attention import TinyAttentionCausalLM
from light_vllm.modeling.runner import ModelRunner


def create_catalog() -> Catalog:
    catalog = Catalog()
    catalog.models.register("tiny-causal-lm", TinyCausalLM.from_spec)
    catalog.models.register("tiny-attention-causal-lm", TinyAttentionCausalLM.from_spec)
    catalog.loaders.register("init", InitModelLoader())
    catalog.loaders.register("state-dict", StateDictModelLoader())
    return catalog


def create_runner(catalog: Catalog | None = None) -> ModelRunner:
    return ModelRunner(catalog or create_catalog())
