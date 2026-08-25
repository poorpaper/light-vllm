"""模型 logits 到 token ID 的采样策略。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

import torch
from torch import Tensor


class SamplingError(RuntimeError):
    """采样输入或输出违反契约时抛出。"""


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """一个生成请求固定使用的采样语义。

    ``temperature=0`` 明确表示 greedy；此时 top-k、top-p 和 seed 不改变结果。
    随机请求的 seed 会在 Engine 注册时补成一个请求私有值。
    """

    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not isfinite(self.temperature)
            or not 0.0 <= self.temperature <= 2.0
        ):
            raise ValueError("temperature must be finite and within [0, 2]")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k <= 0):
            raise ValueError("top_k must be a positive integer")
        if (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not isfinite(self.top_p)
            or not 0.0 < self.top_p <= 1.0
        ):
            raise ValueError("top_p must be finite and within (0, 1]")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**63):
            raise ValueError("seed must be a non-negative 63-bit integer")
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "top_p", float(self.top_p))

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0


@dataclass(frozen=True, slots=True)
class SamplingMetadata:
    """一行 logits 对应的请求及输出位置。"""

    request_id: str
    params: SamplingParams
    output_position: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("sampling request_id must not be empty")
        if type(self.output_position) is not int or self.output_position < 0:
            raise ValueError("sampling output_position must be a non-negative integer")


class Sampler(Protocol):
    """从连续 logits 行选择 token；metadata 与行一一对应。"""

    def sample(
        self,
        logits: Tensor,
        metadata: tuple[SamplingMetadata, ...] | None = None,
    ) -> tuple[int, ...]: ...


def _validate_logits(
    logits: Tensor,
    metadata: tuple[SamplingMetadata, ...] | None,
) -> tuple[SamplingMetadata, ...] | None:
    if logits.ndim != 2:
        raise SamplingError("sampler logits must have shape [batch, vocabulary]")
    if logits.shape[0] == 0 or logits.shape[1] == 0:
        raise SamplingError("sampler logits must not be empty")
    if metadata is not None:
        metadata = tuple(metadata)
        if len(metadata) != logits.shape[0]:
            raise SamplingError("sampling metadata must match the logits rows")
    return metadata


class GreedySampler:
    """为每一行选择分数最高的 token。"""

    def sample(
        self,
        logits: Tensor,
        metadata: tuple[SamplingMetadata, ...] | None = None,
    ) -> tuple[int, ...]:
        _validate_logits(logits, metadata)
        return tuple(int(token_id) for token_id in logits.argmax(dim=-1).tolist())


def _position_seed(seed: int, output_position: int) -> int:
    """从请求 seed 和输出位置得到稳定、与 batch 无关的子 seed。"""

    mixed = seed ^ ((output_position + 1) * 0x9E3779B97F4A7C15)
    mixed ^= mixed >> 30
    mixed *= 0xBF58476D1CE4E5B9
    mixed ^= mixed >> 27
    return mixed & ((1 << 63) - 1)


class ConfigurableSampler:
    """支持 temperature、top-k、top-p 和逐请求固定 seed 的采样器。"""

    def sample(
        self,
        logits: Tensor,
        metadata: tuple[SamplingMetadata, ...] | None = None,
    ) -> tuple[int, ...]:
        metadata = _validate_logits(logits, metadata)
        if metadata is None:
            return tuple(int(token_id) for token_id in logits.argmax(dim=-1).tolist())

        sampled: list[int] = []
        for row, item in zip(logits, metadata, strict=True):
            params = item.params
            if params.is_greedy:
                sampled.append(int(row.argmax().item()))
                continue
            if params.top_k is not None and params.top_k > row.shape[0]:
                raise SamplingError("top_k must not exceed the model vocabulary size")

            scores = row.float()
            if not bool(torch.isfinite(scores).all()):
                raise SamplingError("sampler logits must contain only finite values")
            # 先减去最大值，避免极小但合法的 temperature 把最高分也放大成 +inf。
            temperature = max(params.temperature, torch.finfo(scores.dtype).tiny)
            scores = (scores - scores.max()) / temperature
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            if params.top_k is not None:
                sorted_scores[params.top_k :] = -torch.inf
            if params.top_p < 1.0:
                probabilities = torch.softmax(sorted_scores, dim=-1)
                remove = probabilities.cumsum(dim=-1) > params.top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)

            probabilities = torch.softmax(sorted_scores, dim=-1)
            if not bool(torch.isfinite(probabilities).all()) or float(probabilities.sum()) <= 0:
                raise SamplingError("sampling probabilities are invalid")
            generator = None
            if params.seed is not None:
                generator = torch.Generator(device=logits.device)
                generator.manual_seed(_position_seed(params.seed, item.output_position))
            sampled_index = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            )
            sampled.append(int(sorted_indices[sampled_index].item()))
        return tuple(sampled)
