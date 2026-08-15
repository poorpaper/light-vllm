"""模型加载、原子替换与固定 generation 会话。"""

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
    """强引用一个已加载模型，reload 不会改变它。"""

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
    """管理当前模型指针，并为请求创建固定模型会话。

    Runner 只负责模型生命周期，不负责请求调度、KV 生命周期或执行拓扑。
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

        # 候选模型先在锁外完整构造；加载失败不会影响当前仍可用的模型。
        candidate = loader.load(spec, factory)
        with self._lock:
            # 只有完整可用的候选模型才能原子替换当前模型并推进 generation。
            self._model = candidate
            self._generation += 1

    def open_session(self) -> ModelSession:
        """固定当前模型和 generation，供一次完整生成持续使用。"""

        with self._lock:
            model = self._model
            generation = self._generation
        if model is None:
            raise ModelNotLoadedError("load a model before opening a session")
        # session 持有模型强引用；后续 reload 只会改变 Runner 的当前指针。
        return _PinnedModelSession(
            generation=generation,
            _model=model,
            kv_cache_spec=_kv_cache_spec(model),
            max_model_tokens=_max_model_tokens(model),
        )
