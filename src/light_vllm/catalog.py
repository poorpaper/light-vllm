from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from light_vllm.contracts import ModelFactory, ModelLoader
from light_vllm.registry import Registry


class Plugin(Protocol):
    def register(self, catalog: Catalog) -> None: ...


@dataclass(slots=True)
class Catalog:
    """Owns the extension points for one runtime instance."""

    models: Registry[ModelFactory] = field(default_factory=lambda: Registry("model"))
    loaders: Registry[ModelLoader] = field(default_factory=lambda: Registry("loader"))

    def install(self, plugin: Plugin) -> None:
        plugin.register(self)
