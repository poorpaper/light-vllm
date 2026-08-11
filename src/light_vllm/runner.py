from __future__ import annotations

from threading import RLock

import torch
from torch import nn

from light_vllm.catalog import Catalog
from light_vllm.models.api import ForwardBatch, ModelNotLoadedError, ModelOutput, ModelSpec


class ModelRunner:
    """管理当前模型，并提供统一的推理入口。"""

    def __init__(self, catalog: Catalog) -> None:
        self._catalog = catalog
        self._model: nn.Module | None = None
        self._generation = 0
        self._lock = RLock()

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def load(self, spec: ModelSpec) -> None:
        factory = self._catalog.models.get(spec.architecture)
        loader = self._catalog.loaders.get(spec.loader)

        candidate = loader.load(spec, factory)
        with self._lock:
            self._model = candidate
            self._generation += 1

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch) -> ModelOutput:
        with self._lock:
            model = self._model
        if model is None:
            raise ModelNotLoadedError("load a model before calling forward")

        output = model(batch)
        if not isinstance(output, ModelOutput):
            raise TypeError("registered models must return ModelOutput")
        return output
