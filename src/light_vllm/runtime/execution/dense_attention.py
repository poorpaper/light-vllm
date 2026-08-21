"""连续 K/V 使用的可读 PyTorch attention 实现。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from light_vllm.modeling.attention.interfaces import AttentionContext, ModelKVCacheSpec
from light_vllm.runtime.execution.interfaces import QueryLayout
from light_vllm.runtime.execution.layout import query_visibility
from light_vllm.runtime.kv_cache import (
    ContiguousKVCacheState,
    ContiguousLayerKV,
    KVCacheError,
)


@dataclass(frozen=True, slots=True)
class DenseAttentionMetadata:
    """dense attention 需要的 query 位置和父链布局。"""

    positions: Tensor
    query_layouts: tuple[QueryLayout, ...]

    def __post_init__(self) -> None:
        layouts = tuple(self.query_layouts)
        if self.positions.ndim != 1:
            raise ValueError("dense attention positions must be one-dimensional")
        if self.positions.dtype != torch.long:
            raise ValueError("dense attention positions must use torch.long")
        if bool(torch.any(self.positions < 0)):
            raise ValueError("dense attention positions must not be negative")
        if not layouts or sum(len(layout) for layout in layouts) != self.positions.shape[0]:
            raise ValueError("dense attention layouts must cover the packed token stream")
        object.__setattr__(self, "query_layouts", layouts)

    @property
    def batch_size(self) -> int:
        return len(self.query_layouts)

    @property
    def query_lengths(self) -> tuple[int, ...]:
        return tuple(len(layout) for layout in self.query_layouts)

    @property
    def query_start_loc(self) -> tuple[int, ...]:
        starts = [0]
        for length in self.query_lengths:
            starts.append(starts[-1] + length)
        return tuple(starts)


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
        if self._metadata.batch_size != 1:
            raise KVCacheError("multiple packed requests cannot share one contiguous cache state")
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
        if past is not None and self._metadata.batch_size != 1:
            raise KVCacheError("one contiguous history cannot serve multiple packed requests")

        output = torch.empty_like(query)
        repeats = layer_spec.num_query_heads // layer_spec.num_kv_heads
        for start, end, layout in zip(
            self._metadata.query_start_loc[:-1],
            self._metadata.query_start_loc[1:],
            self._metadata.query_layouts,
            strict=True,
        ):
            query_row = query[start:end]
            keys = key[start:end]
            values = value[start:end]
            past_length = 0
            if past is not None:
                self._validate_past(past, key, layer_spec)
                past_length = past.keys.shape[1]
                keys = torch.cat((past.keys[0], keys), dim=0)
                values = torch.cat((past.values[0], values), dim=0)
            if repeats > 1:
                keys = keys.repeat_interleave(repeats, dim=1)
                values = values.repeat_interleave(repeats, dim=1)
            scores = torch.einsum("qhd,khd->hqk", query_row, keys) * scale
            visibility = torch.tensor(
                query_visibility(layout),
                dtype=torch.bool,
                device=query.device,
            )
            if past_length:
                visibility = torch.cat(
                    (
                        torch.ones(
                            (end - start, past_length), dtype=torch.bool, device=query.device
                        ),
                        visibility,
                    ),
                    dim=1,
                )
            scores = scores.masked_fill(~visibility.unsqueeze(0), torch.finfo(scores.dtype).min)
            probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
            output[start:end] = torch.einsum("hqk,khd->qhd", probabilities, values)

        # 连续缓存仍按单请求保存 batch 维；模型与 attention 接口保持 token-major。
        self._updates[layer_id] = ContiguousLayerKV(
            keys=key.unsqueeze(0),
            values=value.unsqueeze(0),
        )
        return output

    def _validate_inputs(self, query, key, value, layer_spec) -> None:
        if query.ndim != 3:
            raise KVCacheError("dense attention query must have three dimensions")
        if query.shape[0] != key.shape[0] or value.shape != key.shape:
            raise KVCacheError("dense attention Q/K/V token dimensions must match")
        if query.shape[0] != self._metadata.positions.shape[0]:
            raise KVCacheError("dense attention metadata does not match the query")
        if self._metadata.positions.device != query.device:
            raise KVCacheError("dense attention positions must use the query device")
        if query.shape[1:] != (layer_spec.num_query_heads, layer_spec.head_size):
            raise KVCacheError("dense attention query shape does not match the layer spec")
        if key.shape[1:] != (layer_spec.num_kv_heads, layer_spec.head_size):
            raise KVCacheError("dense attention K/V shape does not match the layer spec")

    def _validate_past(self, past, key, layer_spec) -> None:
        expected_tail = (layer_spec.num_kv_heads, layer_spec.head_size)
        if past.keys.shape[0] != 1:
            raise KVCacheError("dense KV cache batch size does not match the query")
        if past.keys.shape[2:] != expected_tail:
            raise KVCacheError("dense KV cache shape does not match the layer spec")
        if past.keys.dtype != key.dtype or past.keys.device != key.device:
            raise KVCacheError("dense KV cache dtype and device must match the new K/V")
