"""模型侧 Tensor Parallel 基础组件。

这里仅描述张量怎样切分、怎样聚合。进程启动、NCCL 初始化和 Rank 生命周期
属于执行层，模型不直接依赖 ``torch.distributed``。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TensorCollectives(Protocol):
    """模型计算实际需要的两种集合通信。"""

    def all_reduce_sum(self, tensor: Tensor) -> Tensor: ...

    def all_gather_last_dim(
        self,
        tensor: Tensor,
        partition_sizes: tuple[int, ...],
    ) -> Tensor: ...


class _SingleProcessCollectives:
    """TP=1 的无通信实现，让模型无需维护两套 forward。"""

    def all_reduce_sum(self, tensor: Tensor) -> Tensor:
        return tensor

    def all_gather_last_dim(
        self,
        tensor: Tensor,
        partition_sizes: tuple[int, ...],
    ) -> Tensor:
        if partition_sizes != (tensor.shape[-1],):
            raise ValueError("single-process gather size must match the local tensor")
        return tensor


@dataclass(frozen=True, slots=True)
class TensorPartition:
    """一个 Rank 在某一完整维度上持有的连续区间。"""

    total_size: int
    rank: int
    world_size: int
    start: int
    end: int
    sizes: tuple[int, ...]

    @property
    def local_size(self) -> int:
        return self.end - self.start


def partition_dimension(total_size: int, rank: int, world_size: int) -> TensorPartition:
    """把一维连续均分；不能整除时，靠前 Rank 多持有一个元素。"""

    if type(total_size) is not int or total_size <= 0:
        raise ValueError("partitioned dimension must be a positive integer")
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("tensor parallel world size must be a positive integer")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("tensor parallel rank must be within the world size")
    base, remainder = divmod(total_size, world_size)
    sizes = tuple(base + (index < remainder) for index in range(world_size))
    if any(size == 0 for size in sizes):
        raise ValueError("tensor parallel world size exceeds the partitioned dimension")
    start = sum(sizes[:rank])
    return TensorPartition(
        total_size=total_size,
        rank=rank,
        world_size=world_size,
        start=start,
        end=start + sizes[rank],
        sizes=sizes,
    )


@dataclass(frozen=True, slots=True)
class TensorParallelContext:
    """模型构造和 forward 共用的 TP 事实。"""

    rank: int = 0
    world_size: int = 1
    collectives: TensorCollectives | None = None

    def __post_init__(self) -> None:
        if type(self.world_size) is not int or self.world_size <= 0:
            raise ValueError("tensor parallel world size must be a positive integer")
        if type(self.rank) is not int or not 0 <= self.rank < self.world_size:
            raise ValueError("tensor parallel rank must be within the world size")
        if self.collectives is None:
            if self.world_size != 1:
                raise ValueError("multi-rank tensor parallelism requires collectives")
            object.__setattr__(self, "collectives", _SingleProcessCollectives())

    def partition(self, total_size: int) -> TensorPartition:
        return partition_dimension(total_size, self.rank, self.world_size)


@dataclass(frozen=True, slots=True)
class TensorShardSpec:
    """checkpoint 中一个完整 tensor 到本 Rank 参数的切片规则。"""

    dimension: int
    start: int
    length: int
    full_size: int

    def __post_init__(self) -> None:
        if type(self.dimension) is not int or self.dimension < 0:
            raise ValueError("shard dimension must be a non-negative integer")
        if type(self.start) is not int or self.start < 0:
            raise ValueError("shard start must be a non-negative integer")
        if type(self.length) is not int or self.length <= 0:
            raise ValueError("shard length must be a positive integer")
        if type(self.full_size) is not int or self.start + self.length > self.full_size:
            raise ValueError("shard must fit within the full tensor dimension")


def _partition_shard(dimension: int, partition: TensorPartition) -> TensorShardSpec:
    return TensorShardSpec(
        dimension=dimension,
        start=partition.start,
        length=partition.local_size,
        full_size=partition.total_size,
    )


def checkpoint_shards(model: nn.Module) -> dict[str, TensorShardSpec]:
    """收集并行层公开的参数切片，不依赖具体模型或参数命名。"""

    result: dict[str, TensorShardSpec] = {}
    for module_name, module in model.named_modules():
        local = getattr(module, "checkpoint_shards", None)
        if local is None:
            continue
        if not isinstance(local, Mapping):
            raise TypeError("module checkpoint_shards must be a mapping")
        for parameter_name, shard in local.items():
            if not isinstance(parameter_name, str) or not parameter_name:
                raise TypeError("checkpoint shard names must be non-empty strings")
            if not isinstance(shard, TensorShardSpec):
                raise TypeError("checkpoint shard values must be TensorShardSpec")
            full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            if full_name in result:
                raise ValueError(f"duplicate checkpoint shard declaration: {full_name!r}")
            result[full_name] = shard
    return result


class ColumnParallelLinear(nn.Linear):
    """按输出维切 Linear；每个 Rank 计算不同的输出列。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        context: TensorParallelContext,
        *,
        bias: bool = True,
        gather_output: bool = False,
        output_partition: TensorPartition | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self._parallel = context
        self._output_partition = output_partition or context.partition(out_features)
        if self._output_partition.total_size != out_features:
            raise ValueError("output partition must cover the linear output dimension")
        self._gather_output = gather_output
        super().__init__(
            in_features,
            self._output_partition.local_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    @property
    def checkpoint_shards(self) -> Mapping[str, TensorShardSpec]:
        shard = _partition_shard(0, self._output_partition)
        result = {"weight": shard}
        if self.bias is not None:
            result["bias"] = shard
        return result

    def gather_output(self, local_output: Tensor) -> Tensor:
        if not self._gather_output:
            return local_output
        assert self._parallel.collectives is not None
        return self._parallel.collectives.all_gather_last_dim(
            local_output,
            self._output_partition.sizes,
        )

    def forward(self, input: Tensor) -> Tensor:
        local_output = F.linear(input, self.weight, self.bias)
        return self.gather_output(local_output)


class RowParallelLinear(nn.Linear):
    """按输入维切 Linear，并把各 Rank 的部分和相加。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        context: TensorParallelContext,
        *,
        bias: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self._parallel = context
        self._input_partition = context.partition(in_features)
        super().__init__(
            self._input_partition.local_size,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    @property
    def checkpoint_shards(self) -> Mapping[str, TensorShardSpec]:
        return {"weight": _partition_shard(1, self._input_partition)}

    def reduce_output(self, local_output: Tensor) -> Tensor:
        assert self._parallel.collectives is not None
        output = self._parallel.collectives.all_reduce_sum(local_output)
        if self.bias is not None:
            output = output + self.bias
        return output

    def forward(self, input: Tensor) -> Tensor:
        local_output = F.linear(input, self.weight, None)
        return self.reduce_output(local_output)


class VocabParallelEmbedding(nn.Embedding):
    """按词表切 Embedding；非本 Rank token 置零后用 AllReduce 合并。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        context: TensorParallelContext,
        padding_idx: int | None = None,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self._parallel = context
        self._vocab_partition = context.partition(num_embeddings)
        local_padding_idx = None
        if (
            padding_idx is not None
            and self._vocab_partition.start <= padding_idx < self._vocab_partition.end
        ):
            local_padding_idx = padding_idx - self._vocab_partition.start
        super().__init__(
            self._vocab_partition.local_size,
            embedding_dim,
            local_padding_idx,
            device=device,
            dtype=dtype,
        )

    @property
    def checkpoint_shards(self) -> Mapping[str, TensorShardSpec]:
        return {"weight": _partition_shard(0, self._vocab_partition)}

    def forward(self, input: Tensor) -> Tensor:
        # TP=1 持有完整词表，直接复用 PyTorch Embedding，避免每步启动
        # token 归属判断、重映射和清零等无意义的 CUDA kernel。
        if self._parallel.world_size == 1:
            return super().forward(input)

        valid = torch.all((input >= 0) & (input < self._vocab_partition.total_size))
        if input.device.type == "cuda":
            torch._assert_async(valid, "token ID exceeds the model vocabulary")
        elif not bool(valid):
            raise ValueError("token ID exceeds the model vocabulary")

        owned = (input >= self._vocab_partition.start) & (input < self._vocab_partition.end)
        local_ids = torch.where(owned, input - self._vocab_partition.start, 0)
        local_output = F.embedding(
            local_ids,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        local_output = local_output.masked_fill(~owned.unsqueeze(-1), 0)
        assert self._parallel.collectives is not None
        return self._parallel.collectives.all_reduce_sum(local_output)


class VocabParallelLinear(ColumnParallelLinear):
    """按词表切 LM Head，并恢复完整 vocabulary logits。"""

    def __init__(
        self,
        in_features: int,
        vocab_size: int,
        context: TensorParallelContext,
        *,
        bias: bool = False,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_features,
            vocab_size,
            context,
            bias=bias,
            gather_output=True,
            device=device,
            dtype=dtype,
        )
