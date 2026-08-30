from __future__ import annotations

import torch
from torch import nn

from light_vllm.modeling.models.interfaces import ModelFactory, ModelSpec


class ModelLoadError(RuntimeError):
    """模型加载失败时抛出。"""


def _require_dense_quantization(spec: ModelSpec, loader_name: str) -> None:
    """没有量化元数据解析能力的 loader 只能构造 Dense 模型。"""

    if spec.quantization not in {"auto", "none"}:
        raise ModelLoadError(
            f"the {loader_name} loader does not support explicit quantization "
            f"'{spec.quantization}'; use the safetensors loader"
        )


def _prepare_for_inference(model: nn.Module, spec: ModelSpec) -> nn.Module:
    model = model.to(device=spec.device, dtype=spec.dtype).eval()
    prepare = getattr(model, "prepare_for_inference", None)
    if prepare is not None:
        if not callable(prepare):
            raise ModelLoadError("model prepare_for_inference must be callable")
        prepare()
    return model


class InitModelLoader:
    """创建新模型，不读取权重文件。"""

    def load(self, spec: ModelSpec, factory: ModelFactory) -> nn.Module:
        _require_dense_quantization(spec, "init")
        return _prepare_for_inference(factory(spec), spec)


class StateDictModelLoader:
    """从可信的本地 PyTorch 权重文件加载模型。"""

    def __init__(self, *, strict: bool = True) -> None:
        self._strict = strict

    def load(self, spec: ModelSpec, factory: ModelFactory) -> nn.Module:
        _require_dense_quantization(spec, "state-dict")
        if spec.weights is None:
            raise ModelLoadError("the state-dict loader requires ModelSpec.weights")

        model = factory(spec)
        state_dict = torch.load(spec.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=self._strict)
        return _prepare_for_inference(model, spec)
