from __future__ import annotations

from typing import Protocol

from torch import nn

from light_vllm.models.api import ModelFactory, ModelSpec


class ModelLoader(Protocol):
    """创建模型、加载权重并准备推理。"""

    def load(self, spec: ModelSpec, factory: ModelFactory) -> nn.Module: ...
