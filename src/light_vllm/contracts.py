from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Everything required to choose and materialize one model."""

    architecture: str
    loader: str = "init"
    model_args: Mapping[str, object] = field(default_factory=dict)
    weights: Path | None = None
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32


@dataclass(frozen=True, slots=True)
class ForwardBatch:
    """The stable input boundary between request preparation and a model."""

    input_ids: Tensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """The minimal output needed by the next runtime layer."""

    logits: Tensor


class ModelFactory(Protocol):
    def __call__(self, spec: ModelSpec) -> nn.Module: ...


class ModelLoader(Protocol):
    def load(self, spec: ModelSpec, factory: ModelFactory) -> nn.Module: ...
