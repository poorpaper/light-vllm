from __future__ import annotations

from light_vllm.catalog import Catalog
from light_vllm.loaders import InitModelLoader, StateDictModelLoader
from light_vllm.models import TinyCausalLM
from light_vllm.runner import ModelRunner


def create_catalog() -> Catalog:
    catalog = Catalog()
    catalog.models.register("tiny-causal-lm", TinyCausalLM.from_spec)
    catalog.loaders.register("init", InitModelLoader())
    catalog.loaders.register("state-dict", StateDictModelLoader())
    return catalog


def create_runner(catalog: Catalog | None = None) -> ModelRunner:
    return ModelRunner(catalog or create_catalog())
