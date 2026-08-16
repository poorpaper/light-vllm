from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor


@dataclass(frozen=True, slots=True)
class AttentionLayerSpec:
    """一个 attention 层的逻辑 K/V 形状。"""

    layer_id: str
    num_query_heads: int
    num_kv_heads: int
    head_size: int

    def __post_init__(self) -> None:
        if not self.layer_id:
            raise ValueError("layer_id must not be empty")
        for name, value in (
            ("num_query_heads", self.num_query_heads),
            ("num_kv_heads", self.num_kv_heads),
            ("head_size", self.head_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")


@dataclass(frozen=True, slots=True)
class ModelKVCacheSpec:
    """模型所有可缓存 attention 层的 KV 形状说明。"""

    layers: tuple[AttentionLayerSpec, ...]

    def __post_init__(self) -> None:
        layers = tuple(self.layers)
        layer_ids = tuple(layer.layer_id for layer in layers)
        if not layers:
            raise ValueError("model KV cache spec must contain at least one layer")
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("attention layer IDs must be unique")
        object.__setattr__(self, "layers", layers)


class AttentionContext(Protocol):
    """模型 attention 层通过这个接口读写本批次的 KV cache。"""

    def forward(
        self,
        layer_id: str,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        scale: float,
    ) -> Tensor:
        """写入本轮 K/V，并返回与 ``query`` 同形状的 attention 结果。"""

        ...
