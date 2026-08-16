"""加载模型，并保证一次生成不会在中途换模型。"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import torch
from torch import nn

from light_vllm.modeling.attention.interfaces import ModelKVCacheSpec
from light_vllm.modeling.catalog import Catalog
from light_vllm.modeling.models.interfaces import (
    ForwardBatch,
    ModelNotLoadedError,
    ModelOutput,
    ModelSession,
    ModelSpec,
)


def _kv_cache_spec(model: nn.Module) -> ModelKVCacheSpec | None:
    spec = getattr(model, "kv_cache_spec", None)
    if spec is not None and not isinstance(spec, ModelKVCacheSpec):
        raise TypeError("model kv_cache_spec must be a ModelKVCacheSpec")
    return spec


def _max_model_tokens(model: nn.Module) -> int | None:
    value = getattr(model, "max_model_tokens", None)
    if value is not None and (type(value) is not int or value <= 0):
        raise TypeError("model max_model_tokens must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class _PinnedModelSession:
    """保存本次执行选中的模型；之后重新加载模型不会影响本次执行。"""

    generation: int
    _model: nn.Module
    kv_cache_spec: ModelKVCacheSpec | None
    max_model_tokens: int | None

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch) -> ModelOutput:
        output = self._model(batch)
        if not isinstance(output, ModelOutput):
            raise TypeError("registered models must return ModelOutput")
        return output


class ModelRunner:
    """管理当前加载的模型，并让每次执行先选定要使用的模型。

    它只负责加载和切换模型，不负责请求调度、KV cache 或设备执行。
    """

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

        # 先把新模型完整加载好；加载失败时，旧模型仍然可以继续服务。
        candidate = loader.load(spec, factory)
        with self._lock:
            # 新模型加载成功后再一次性替换旧模型，并递增模型版本号。
            self._model = candidate
            self._generation += 1

    def open_session(self) -> ModelSession:
        """取得当前模型，供一次完整生成持续使用。"""

        with self._lock:
            model = self._model
            generation = self._generation
        if model is None:
            raise ModelNotLoadedError("load a model before opening a session")
        # 返回值保存当前模型；之后 reload 不会让正在执行的请求换模型。
        return _PinnedModelSession(
            generation=generation,
            _model=model,
            kv_cache_spec=_kv_cache_spec(model),
            max_model_tokens=_max_model_tokens(model),
        )
