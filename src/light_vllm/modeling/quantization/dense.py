"""不量化的 LinearMethod，实现现有 Dense 推理快路径。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from light_vllm.modeling.quantization.interfaces import DirectLinear
from light_vllm.modeling.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelContext,
    TensorPartition,
)


class DenseLinearMethod:
    """用 PyTorch Dense 参数和预转置权重执行 Linear。"""

    def create_column(
        self,
        in_features: int,
        out_features: int,
        parallel: TensorParallelContext,
        *,
        prefix: str,
        bias: bool = True,
        gather_output: bool = False,
        output_partition: TensorPartition | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> ColumnParallelLinear:
        del prefix
        return ColumnParallelLinear(
            in_features,
            out_features,
            parallel,
            bias=bias,
            gather_output=gather_output,
            output_partition=output_partition,
            device=device,
            dtype=dtype,
        )

    def create_row(
        self,
        in_features: int,
        out_features: int,
        parallel: TensorParallelContext,
        *,
        prefix: str,
        bias: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> RowParallelLinear:
        del prefix
        return RowParallelLinear(
            in_features,
            out_features,
            parallel,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def prepare_local(self, layer: nn.Module) -> DirectLinear:
        if not isinstance(layer, (ColumnParallelLinear, RowParallelLinear)):
            raise TypeError("DenseLinearMethod can only prepare dense parallel linear layers")
        return DirectLinear(weight_t=layer.weight.t(), bias=layer.bias)

    def prepare_merged(self, layers: Sequence[nn.Module]) -> DirectLinear:
        if not layers:
            raise ValueError("cannot merge an empty linear layer sequence")
        if any(not isinstance(layer, ColumnParallelLinear) for layer in layers):
            raise TypeError("DenseLinearMethod only merges dense column-parallel layers")
        dense_layers = tuple(layer for layer in layers if isinstance(layer, ColumnParallelLinear))
        with torch.no_grad():
            weight = torch.cat(tuple(layer.weight for layer in dense_layers), dim=0).contiguous()
            biases = tuple(layer.bias for layer in dense_layers)
            if all(bias is None for bias in biases):
                bias = None
            elif any(bias is None for bias in biases):
                raise ValueError("merged dense layers must either all have bias or all omit it")
            else:
                bias = torch.cat(tuple(bias for bias in biases if bias is not None), dim=0)
                bias = bias.contiguous()
        return DirectLinear(weight_t=weight.t(), bias=bias)
