"""连续 K/V 使用的可读 PyTorch attention 实现。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from light_vllm.modeling.attention.interfaces import AttentionContext, ModelKVCacheSpec
from light_vllm.runtime.kv_cache import (
    ContiguousKVCacheState,
    ContiguousLayerKV,
    KVCacheError,
)


@dataclass(frozen=True, slots=True)
class DenseAttentionMetadata:
    """dense attention 需要的 query 位置和有效长度。"""

    positions: Tensor
    query_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        lengths = tuple(self.query_lengths)
        if self.positions.ndim != 2:
            raise ValueError("dense attention positions must have two dimensions")
        if self.positions.dtype != torch.long:
            raise ValueError("dense attention positions must use torch.long")
        if bool(torch.any(self.positions < 0)):
            raise ValueError("dense attention positions must not be negative")
        if len(lengths) != self.positions.shape[0]:
            raise ValueError("dense attention needs one query length per batch row")
        if any(
            type(length) is not int or length <= 0 or length > self.positions.shape[1]
            for length in lengths
        ):
            raise ValueError("dense attention query lengths must fit the padded width")
        object.__setattr__(self, "query_lengths", lengths)

    @property
    def batch_size(self) -> int:
        return self.positions.shape[0]

    @property
    def query_width(self) -> int:
        return self.positions.shape[1]


class TorchDenseAttention(AttentionContext):
    """在连续历史 K/V 上直接计算 causal attention。

    模型只提供本轮 Q/K/V。这个上下文读取已有连续缓存、完成 attention，
    并暂存各层的新 K/V；模型全部成功后，Step Handler 再一次性追加缓存。
    """

    def __init__(
        self,
        model_spec: ModelKVCacheSpec,
        metadata: DenseAttentionMetadata,
        past: ContiguousKVCacheState | None = None,
    ) -> None:
        self._model_spec = model_spec
        self._metadata = metadata
        self._layer_specs = {layer.layer_id: layer for layer in model_spec.layers}
        self._past_by_layer: dict[str, ContiguousLayerKV] = {}
        if past is not None:
            if len(past.layers) != len(model_spec.layers):
                raise KVCacheError("dense KV cache layer count does not match the model spec")
            self._past_by_layer = {
                spec.layer_id: layer
                for spec, layer in zip(model_spec.layers, past.layers, strict=True)
            }
        self._updates: dict[str, ContiguousLayerKV] = {}

    @property
    def layer_ids(self) -> frozenset[str]:
        return frozenset(self._updates)

    @property
    def cache_updates(self) -> ContiguousKVCacheState:
        """按模型层顺序返回本轮 K/V；缺层时拒绝产生部分更新。"""

        expected = frozenset(self._layer_specs)
        if self.layer_ids != expected:
            raise KVCacheError("model did not execute every configured dense attention layer")
        if any(length != self._metadata.query_width for length in self._metadata.query_lengths):
            raise KVCacheError("padded dense batches cannot be appended to one contiguous cache")
        return ContiguousKVCacheState(
            layers=tuple(self._updates[layer.layer_id] for layer in self._model_spec.layers)
        )

    def forward(
        self,
        layer_id: str,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        scale: float,
    ) -> Tensor:
        if layer_id in self._updates:
            raise KVCacheError(f"dense attention layer {layer_id!r} ran more than once")
        try:
            layer_spec = self._layer_specs[layer_id]
        except KeyError as exc:
            raise KVCacheError(f"dense attention layer {layer_id!r} was not configured") from exc
        self._validate_inputs(query, key, value, layer_spec)

        past = self._past_by_layer.get(layer_id)
        past_length = 0
        keys = key
        values = value
        if past is not None:
            self._validate_past(past, key, layer_spec)
            past_length = past.keys.shape[1]
            keys = torch.cat((past.keys, key), dim=1)
            values = torch.cat((past.values, value), dim=1)

        repeats = layer_spec.num_query_heads // layer_spec.num_kv_heads
        if repeats > 1:
            keys = keys.repeat_interleave(repeats, dim=2)
            values = values.repeat_interleave(repeats, dim=2)

        scores = torch.einsum("bqhd,bkhd->bhqk", query, keys) * scale
        mask = self._causal_mask(past_length, query.device)
        scores = scores.masked_fill(~mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)

        # 这里只暂存引用；模型完整成功后，连续缓存再统一校验并复制所有层。
        self._updates[layer_id] = ContiguousLayerKV(keys=key, values=value)
        return torch.einsum("bhqk,bkhd->bqhd", probabilities, values)

    def _causal_mask(self, past_length: int, device: torch.device) -> Tensor:
        positions = self._metadata.positions
        past_positions = torch.arange(
            past_length,
            dtype=torch.long,
            device=device,
        ).expand(self._metadata.batch_size, -1)
        key_positions = torch.cat((past_positions, positions), dim=1)
        query_offsets = torch.arange(self._metadata.query_width, device=device)
        lengths = torch.tensor(self._metadata.query_lengths, device=device).unsqueeze(1)
        valid_new_keys = query_offsets.unsqueeze(0) < lengths
        valid_keys = torch.cat(
            (
                torch.ones(
                    (self._metadata.batch_size, past_length),
                    dtype=torch.bool,
                    device=device,
                ),
                valid_new_keys,
            ),
            dim=1,
        )
        # padding 位置可以经过模型其他层，但不能成为有效 query 的历史。
        causal = key_positions.unsqueeze(1) <= positions.unsqueeze(2)
        return causal & valid_keys.unsqueeze(1)

    def _validate_inputs(self, query, key, value, layer_spec) -> None:
        if query.ndim != 4:
            raise KVCacheError("dense attention query must have four dimensions")
        if query.shape[:2] != key.shape[:2] or value.shape != key.shape:
            raise KVCacheError("dense attention Q/K/V batch and query dimensions must match")
        if query.shape[:2] != (
            self._metadata.batch_size,
            self._metadata.query_width,
        ):
            raise KVCacheError("dense attention metadata does not match the query")
        if self._metadata.positions.device != query.device:
            raise KVCacheError("dense attention positions must use the query device")
        if query.shape[2:] != (layer_spec.num_query_heads, layer_spec.head_size):
            raise KVCacheError("dense attention query shape does not match the layer spec")
        if key.shape[2:] != (layer_spec.num_kv_heads, layer_spec.head_size):
            raise KVCacheError("dense attention K/V shape does not match the layer spec")

    def _validate_past(self, past, key, layer_spec) -> None:
        expected_tail = (layer_spec.num_kv_heads, layer_spec.head_size)
        if past.keys.shape[0] != self._metadata.batch_size:
            raise KVCacheError("dense KV cache batch size does not match the query")
        if past.keys.shape[2:] != expected_tail:
            raise KVCacheError("dense KV cache shape does not match the layer spec")
        if past.keys.dtype != key.dtype or past.keys.device != key.device:
            raise KVCacheError("dense KV cache dtype and device must match the new K/V")
