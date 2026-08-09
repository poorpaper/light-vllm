from __future__ import annotations

import torch
from torch import nn

from light_vllm.contracts import ModelFactory, ModelSpec


class ModelLoadError(RuntimeError):
    """Raised when a loader cannot materialize a valid model."""


def _prepare_for_inference(model: nn.Module, spec: ModelSpec) -> nn.Module:
    return model.to(device=spec.device, dtype=spec.dtype).eval()


class InitModelLoader:
    """Construct a model from its factory without reading a checkpoint."""

    def load(self, spec: ModelSpec, factory: ModelFactory) -> nn.Module:
        return _prepare_for_inference(factory(spec), spec)


class StateDictModelLoader:
    """Load a trusted local PyTorch state dict into a registered model."""

    def __init__(self, *, strict: bool = True) -> None:
        self._strict = strict

    def load(self, spec: ModelSpec, factory: ModelFactory) -> nn.Module:
        if spec.weights is None:
            raise ModelLoadError("the state-dict loader requires ModelSpec.weights")

        model = factory(spec)
        state_dict = torch.load(spec.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=self._strict)
        return _prepare_for_inference(model, spec)
