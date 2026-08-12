from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from light_vllm.modeling.loaders.interfaces import ModelLoader
from light_vllm.modeling.models.interfaces import ModelFactory
from light_vllm.modeling.registry import Registry


class Plugin(Protocol):
    def register(self, catalog: Catalog) -> None: ...


@dataclass(slots=True)
class Catalog:
    """保存当前运行时可用的模型和加载器。"""

    models: Registry[ModelFactory] = field(default_factory=lambda: Registry("model"))
    loaders: Registry[ModelLoader] = field(default_factory=lambda: Registry("loader"))

    def install(self, plugin: Plugin) -> None:
        plugin.register(self)
