"""AutoAWQ GEMM 兼容的 W4A16 权重与 PyTorch 正确性后端。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from light_vllm.modeling.quantization.dense import DenseLinearMethod
from light_vllm.modeling.quantization.interfaces import (
    DirectLinear,
    LinearOperation,
    PreparedLinear,
)
from light_vllm.modeling.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelContext,
    TensorPartition,
    TensorShardSpec,
)

_AWQ_BITS = 4
_AWQ_PACK_FACTOR = 32 // _AWQ_BITS
# AutoAWQ GEMM 并非按自然列顺序塞入 int32。比如原始列 0..7 会按
# 0,2,4,6,1,3,5,7 放进 bit 0..31；读取时必须应用逆置换，否则 tensor
# shape 全对但每 8 列都会静默错位。
_AWQ_PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
_AWQ_UNPACK_ORDER = tuple(_AWQ_PACK_ORDER.index(index) for index in range(_AWQ_PACK_FACTOR))


@dataclass(frozen=True, slots=True)
class AWQConfig:
    """运行时支持的 AWQ checkpoint 子集。

    第一版刻意只接受行业最常见的 AutoAWQ GEMM、非对称 INT4。明确拒绝
    GEMV、Marlin 私有重排和无 zero-point 变体，避免“能加载但数值错误”。
    """

    bits: int = _AWQ_BITS
    group_size: int = 128
    zero_point: bool = True
    version: str = "gemm"
    modules_to_not_convert: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.bits != _AWQ_BITS:
            raise ValueError("AWQ currently requires 4-bit weights")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("AWQ group_size must be positive or -1")
        if not self.zero_point:
            raise ValueError("AWQ currently requires asymmetric zero points")
        if self.version.lower() != "gemm":
            raise ValueError("AWQ currently supports only the AutoAWQ GEMM layout")
        if any(not isinstance(name, str) or not name for name in self.modules_to_not_convert):
            raise ValueError("AWQ modules_to_not_convert must contain non-empty strings")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> AWQConfig:
        """兼容 Hugging Face 与 AutoAWQ 使用的两套字段名。"""

        bits = value.get("bits", value.get("w_bit", _AWQ_BITS))
        group_size = value.get("group_size", value.get("q_group_size", 128))
        zero_point = value.get("zero_point", True)
        version = value.get("version", "gemm")
        skipped = value.get("modules_to_not_convert", ())
        if type(bits) is not int:
            raise ValueError("AWQ bits must be an integer")
        if type(group_size) is not int:
            raise ValueError("AWQ group_size must be an integer")
        if type(zero_point) is not bool:
            raise ValueError("AWQ zero_point must be a boolean")
        if not isinstance(version, str):
            raise ValueError("AWQ version must be a string")
        if skipped is None:
            skipped = ()
        if not isinstance(skipped, (list, tuple)):
            raise ValueError("AWQ modules_to_not_convert must be a list")
        return cls(
            bits=bits,
            group_size=group_size,
            zero_point=zero_point,
            version=version.lower(),
            modules_to_not_convert=tuple(skipped),
        )

    def skips(self, prefix: str) -> bool:
        return any(name in prefix for name in self.modules_to_not_convert)


def _shard(
    dimension: int,
    *,
    start: int,
    length: int,
    full_size: int,
) -> TensorShardSpec:
    return TensorShardSpec(
        dimension=dimension,
        start=start,
        length=length,
        full_size=full_size,
    )


def _validate_packable_output(partition: TensorPartition) -> None:
    if partition.start % _AWQ_PACK_FACTOR or partition.local_size % _AWQ_PACK_FACTOR:
        raise ValueError("AWQ tensor-parallel output partitions must align to 8 packed values")


class AWQColumnParallelLinear(nn.Module):
    """输出维切分的 AutoAWQ GEMM Linear。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        parallel: TensorParallelContext,
        config: AWQConfig,
        *,
        bias: bool = True,
        gather_output: bool = False,
        output_partition: TensorPartition | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._parallel = parallel
        self._output_partition = output_partition or parallel.partition(out_features)
        if self._output_partition.total_size != out_features:
            raise ValueError("output partition must cover the AWQ linear output dimension")
        _validate_packable_output(self._output_partition)
        group_size = in_features if config.group_size == -1 else config.group_size
        if in_features % group_size:
            raise ValueError("AWQ input size must be divisible by group_size")
        self.in_features = in_features
        self.out_features = self._output_partition.local_size
        self.group_size = group_size
        self._gather_output = gather_output
        local_packed_output = self.out_features // _AWQ_PACK_FACTOR
        num_groups = in_features // group_size
        self.register_buffer(
            "qweight",
            torch.empty(in_features, local_packed_output, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "qzeros",
            torch.empty(num_groups, local_packed_output, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "scales",
            torch.empty(num_groups, self.out_features, dtype=dtype, device=device),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.empty(self.out_features, dtype=dtype, device=device),
            )
        else:
            self.bias = None

    @property
    def checkpoint_shards(self) -> Mapping[str, TensorShardSpec]:
        partition = self._output_partition
        packed_start = partition.start // _AWQ_PACK_FACTOR
        packed_length = partition.local_size // _AWQ_PACK_FACTOR
        packed_total = partition.total_size // _AWQ_PACK_FACTOR
        result = {
            "qweight": _shard(
                1,
                start=packed_start,
                length=packed_length,
                full_size=packed_total,
            ),
            "qzeros": _shard(
                1,
                start=packed_start,
                length=packed_length,
                full_size=packed_total,
            ),
            "scales": _shard(
                1,
                start=partition.start,
                length=partition.local_size,
                full_size=partition.total_size,
            ),
        }
        if self.bias is not None:
            result["bias"] = _shard(
                0,
                start=partition.start,
                length=partition.local_size,
                full_size=partition.total_size,
            )
        return result

    def gather_output(self, local_output: Tensor) -> Tensor:
        if not self._gather_output:
            return local_output
        assert self._parallel.collectives is not None
        return self._parallel.collectives.all_gather_last_dim(
            local_output,
            self._output_partition.sizes,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        local_output = _torch_awq_linear(
            inputs,
            self.qweight,
            self.qzeros,
            self.scales,
            self.group_size,
            self.bias,
        )
        return self.gather_output(local_output)


class AWQRowParallelLinear(nn.Module):
    """输入维切分的 AutoAWQ GEMM Linear。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        parallel: TensorParallelContext,
        config: AWQConfig,
        *,
        bias: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if out_features % _AWQ_PACK_FACTOR:
            raise ValueError("AWQ output size must align to 8 packed values")
        self._parallel = parallel
        self._input_partition = parallel.partition(in_features)
        self._global_group_size = in_features if config.group_size == -1 else config.group_size
        if in_features % self._global_group_size:
            raise ValueError("AWQ input size must be divisible by group_size")
        if config.group_size != -1 and (
            self._input_partition.start % config.group_size
            or self._input_partition.local_size % config.group_size
        ):
            raise ValueError("AWQ row-parallel input partitions must align to group_size")
        self.in_features = self._input_partition.local_size
        self.out_features = out_features
        # group_size=-1 表示全输入只有一组；每个 Rank 复制同一 scale/zero，
        # 再对自己的输入分片使用这一组参数。
        self.group_size = self.in_features if config.group_size == -1 else config.group_size
        num_groups = 1 if config.group_size == -1 else self.in_features // self.group_size
        packed_output = out_features // _AWQ_PACK_FACTOR
        self.register_buffer(
            "qweight",
            torch.empty(self.in_features, packed_output, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "qzeros",
            torch.empty(num_groups, packed_output, dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "scales",
            torch.empty(num_groups, out_features, dtype=dtype, device=device),
        )
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.bias = None

    @property
    def checkpoint_shards(self) -> Mapping[str, TensorShardSpec]:
        partition = self._input_partition
        result = {
            "qweight": _shard(
                0,
                start=partition.start,
                length=partition.local_size,
                full_size=partition.total_size,
            )
        }
        if self._global_group_size == partition.total_size:
            # per-column（group_size=-1）的 scale/zero 对所有 Rank 都相同。
            return result
        group_start = partition.start // self._global_group_size
        group_length = partition.local_size // self._global_group_size
        total_groups = partition.total_size // self._global_group_size
        group_shard = _shard(
            0,
            start=group_start,
            length=group_length,
            full_size=total_groups,
        )
        result["qzeros"] = group_shard
        result["scales"] = group_shard
        return result

    def reduce_output(self, local_output: Tensor) -> Tensor:
        assert self._parallel.collectives is not None
        output = self._parallel.collectives.all_reduce_sum(local_output)
        if self.bias is not None:
            output = output + self.bias
        return output

    def forward(self, inputs: Tensor) -> Tensor:
        local_output = _torch_awq_linear(
            inputs,
            self.qweight,
            self.qzeros,
            self.scales,
            self.group_size,
            None,
        )
        return self.reduce_output(local_output)


def unpack_awq(packed: Tensor) -> Tensor:
    """按 AutoAWQ GEMM 列顺序把 int32 展开成 uint4 值。"""

    if packed.dtype != torch.int32 or packed.ndim != 2:
        raise ValueError("packed AWQ tensors must be two-dimensional int32")
    shifts = torch.arange(0, 32, _AWQ_BITS, dtype=torch.int32, device=packed.device)
    values = torch.bitwise_right_shift(packed.unsqueeze(-1), shifts) & 0xF
    values = values[..., _AWQ_UNPACK_ORDER]
    return values.reshape(packed.shape[0], packed.shape[1] * _AWQ_PACK_FACTOR)


def pack_awq(values: Tensor) -> Tensor:
    """把自然列顺序的 uint4 值打包成 AutoAWQ GEMM int32。"""

    if values.ndim != 2 or values.shape[1] % _AWQ_PACK_FACTOR:
        raise ValueError("AWQ values must be a 2D tensor with a multiple-of-8 width")
    if values.is_floating_point() or values.is_complex():
        raise ValueError("AWQ packed values must use an integer dtype")
    if values.numel() and (bool(torch.any(values < 0)) or bool(torch.any(values > 15))):
        raise ValueError("AWQ packed values must stay within uint4 range")
    groups = values.to(torch.int32).reshape(values.shape[0], -1, _AWQ_PACK_FACTOR)
    ordered = groups[..., _AWQ_PACK_ORDER]
    shifts = torch.arange(0, 32, _AWQ_BITS, dtype=torch.int32, device=values.device)
    return torch.sum(ordered << shifts, dim=-1, dtype=torch.int32)


@dataclass(frozen=True, slots=True)
class PackedAWQWeight:
    """可直接写入 AutoAWQ checkpoint 的三个常驻张量。"""

    qweight: Tensor
    qzeros: Tensor
    scales: Tensor


@dataclass(frozen=True, slots=True)
class QuantizedAWQWeight:
    """一个 Dense 权重量化后的 checkpoint 张量及其伪量化基线。"""

    qweight: Tensor
    qzeros: Tensor
    scales: Tensor
    dequantized: Tensor


def _quantize_awq_groups(
    weight: Tensor,
    group_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """按输入维分组，返回 ``(INT4 值, zero, scale)``。

    搜索阶段和 checkpoint 导出必须共用这套非对称量化规则；scale 先落到
    权重 dtype，再据此计算 zero 和 INT4，保证伪量化结果与最终文件一致。
    """

    width = weight.shape[-1]
    effective_group_size = width if group_size == -1 else group_size
    if effective_group_size <= 0 or width % effective_group_size:
        raise ValueError("AWQ group_size must divide the weight input dimension")
    grouped = weight.float().reshape(*weight.shape[:-1], -1, effective_group_size)
    minimum = grouped.amin(dim=-1, keepdim=True)
    maximum = grouped.amax(dim=-1, keepdim=True)
    scales = ((maximum - minimum).clamp_min(1e-5) / 15).to(weight.dtype)
    float_scales = scales.float()
    zeros = torch.round(-minimum / float_scales).clamp_(0, 15).to(torch.int32)
    quantized = torch.round(grouped / float_scales + zeros).clamp_(0, 15).to(torch.int32)
    return quantized, zeros, scales


def pack_awq_checkpoint_weight(weight: Tensor, group_size: int) -> PackedAWQWeight:
    """按输出通道、输入分组生成 AutoAWQ checkpoint 张量。

    输入沿用 ``nn.Linear.weight=[out,in]``；checkpoint 则保存 AutoAWQ GEMM
    使用的 ``qweight=[in,out/8]``。导出路径不构造完整反量化矩阵，避免在
    大模型 CPU 内存中无意义地常驻第二份权重。
    """

    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("AWQ source weight must be a two-dimensional floating tensor")
    out_features, in_features = weight.shape
    effective_group_size = in_features if group_size == -1 else group_size
    if effective_group_size <= 0 or in_features % effective_group_size:
        raise ValueError("AWQ group_size must divide the weight input dimension")
    if out_features % _AWQ_PACK_FACTOR:
        raise ValueError("AWQ weight output dimension must be divisible by 8")

    quantized, zeros, scales = _quantize_awq_groups(weight, effective_group_size)
    natural_qweight = quantized.reshape_as(weight).t().contiguous()
    natural_qzeros = zeros.squeeze(-1).t().contiguous()
    checkpoint_scales = scales.squeeze(-1).t().contiguous()
    return PackedAWQWeight(
        qweight=pack_awq(natural_qweight),
        qzeros=pack_awq(natural_qzeros),
        scales=checkpoint_scales,
    )


def quantize_awq_weight(weight: Tensor, group_size: int) -> QuantizedAWQWeight:
    """量化并返回反量化矩阵，供数值基线和小型测试 fixture 使用。"""

    packed = pack_awq_checkpoint_weight(weight, group_size)
    dequantized = dequantize_awq(
        packed.qweight,
        packed.qzeros,
        packed.scales,
        weight.shape[1] if group_size == -1 else group_size,
    ).t()
    return QuantizedAWQWeight(
        qweight=packed.qweight,
        qzeros=packed.qzeros,
        scales=packed.scales,
        dequantized=dequantized,
    )


def dequantize_awq(
    qweight: Tensor,
    qzeros: Tensor,
    scales: Tensor,
    group_size: int,
) -> Tensor:
    """还原 ``[in_features, out_features]`` 权重，作为 kernel 数值基线。"""

    if group_size <= 0 or qweight.shape[0] % group_size:
        raise ValueError("AWQ group_size must divide the packed input dimension")
    weights = unpack_awq(qweight)
    zeros = unpack_awq(qzeros)
    expected_groups = qweight.shape[0] // group_size
    expected_shape = (expected_groups, weights.shape[1])
    if zeros.shape != expected_shape or scales.shape != expected_shape:
        raise ValueError("AWQ qzeros/scales do not match qweight and group_size")
    group_ids = torch.arange(qweight.shape[0], device=qweight.device) // group_size
    return (weights - zeros[group_ids]).to(scales.dtype) * scales[group_ids]


def _torch_awq_linear(
    inputs: Tensor,
    qweight: Tensor,
    qzeros: Tensor,
    scales: Tensor,
    group_size: int,
    bias: Tensor | None,
) -> Tensor:
    weights = dequantize_awq(qweight, qzeros, scales, group_size)
    output = torch.matmul(inputs, weights.to(inputs.dtype))
    if bias is not None:
        output = output + bias
    return output


class AWQScheme(Protocol):
    """把同一份 AutoAWQ checkpoint 权重交给一个具体计算后端。

    ``AWQLinearMethod`` 管理层类型和 TP 切片；Scheme 只在模型加载完成后
    准备算子。这样新增 Marlin、Triton 或其他平台实现时，不需要改模型结构、
    loader，也不会在逐 token 热路径里反复判断 backend。
    """

    @property
    def name(self) -> str: ...

    def prepare(
        self,
        qweight: Tensor,
        qzeros: Tensor,
        scales: Tensor,
        group_size: int,
        bias: Tensor | None,
    ) -> LinearOperation: ...


@dataclass(frozen=True, slots=True)
class _ConcatenatedOperation:
    operations: tuple[PreparedLinear, ...]

    def __call__(self, inputs: Tensor) -> Tensor:
        outputs = []
        for operation in self.operations:
            if isinstance(operation, DirectLinear):
                if operation.bias is None:
                    outputs.append(torch.mm(inputs, operation.weight_t))
                else:
                    outputs.append(torch.addmm(operation.bias, inputs, operation.weight_t))
            else:
                outputs.append(operation(inputs))
        return torch.cat(outputs, dim=-1)


def _share_merged_awq_storage(
    layers: Sequence[AWQColumnParallelLinear],
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """合并权重后把子层 buffer 重新绑定为 view，不保留第二份 W4 权重。"""

    qweight = torch.cat(tuple(layer.qweight for layer in layers), dim=1).contiguous()
    qzeros = torch.cat(tuple(layer.qzeros for layer in layers), dim=1).contiguous()
    scales = torch.cat(tuple(layer.scales for layer in layers), dim=1).contiguous()
    biases = tuple(layer.bias for layer in layers)
    if all(bias is None for bias in biases):
        bias = None
    elif any(bias is None for bias in biases):
        raise ValueError("merged AWQ layers must either all have bias or all omit it")
    else:
        bias = torch.cat(tuple(value for value in biases if value is not None), dim=0)
        bias = bias.contiguous()

    packed_offset = 0
    output_offset = 0
    for layer in layers:
        packed_width = layer.qweight.shape[1]
        output_width = layer.scales.shape[1]
        layer.qweight = qweight[:, packed_offset : packed_offset + packed_width]
        layer.qzeros = qzeros[:, packed_offset : packed_offset + packed_width]
        layer.scales = scales[:, output_offset : output_offset + output_width]
        if bias is not None:
            layer.bias = bias[output_offset : output_offset + output_width]
        packed_offset += packed_width
        output_offset += output_width
    return qweight, qzeros, scales, bias


class AWQLinearMethod:
    """创建 AutoAWQ 权重，并把 kernel 选择委托给一个 Scheme。"""

    def __init__(self, config: AWQConfig, scheme: AWQScheme) -> None:
        self.config = config
        self.scheme = scheme
        self._dense = DenseLinearMethod()

    def _operation(
        self,
        qweight: Tensor,
        qzeros: Tensor,
        scales: Tensor,
        group_size: int,
        bias: Tensor | None,
    ) -> PreparedLinear:
        return self.scheme.prepare(qweight, qzeros, scales, group_size, bias)

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
    ) -> nn.Module:
        if self.config.skips(prefix):
            return self._dense.create_column(
                in_features,
                out_features,
                parallel,
                prefix=prefix,
                bias=bias,
                gather_output=gather_output,
                output_partition=output_partition,
                device=device,
                dtype=dtype,
            )
        return AWQColumnParallelLinear(
            in_features,
            out_features,
            parallel,
            self.config,
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
    ) -> nn.Module:
        if self.config.skips(prefix):
            return self._dense.create_row(
                in_features,
                out_features,
                parallel,
                prefix=prefix,
                bias=bias,
                device=device,
                dtype=dtype,
            )
        return AWQRowParallelLinear(
            in_features,
            out_features,
            parallel,
            self.config,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def prepare_local(self, layer: nn.Module) -> PreparedLinear:
        if isinstance(layer, (ColumnParallelLinear, RowParallelLinear)):
            return self._dense.prepare_local(layer)
        if isinstance(layer, AWQColumnParallelLinear):
            bias = layer.bias
        elif isinstance(layer, AWQRowParallelLinear):
            # Row Parallel 的 bias 必须在 AllReduce 之后只加一次。
            bias = None
        else:
            raise TypeError("AWQLinearMethod received an unsupported linear layer")
        return self._operation(
            layer.qweight,
            layer.qzeros,
            layer.scales,
            layer.group_size,
            bias,
        )

    def prepare_merged(self, layers: Sequence[nn.Module]) -> PreparedLinear:
        if not layers:
            raise ValueError("cannot merge an empty linear layer sequence")
        if all(isinstance(layer, ColumnParallelLinear) for layer in layers):
            return self._dense.prepare_merged(layers)
        if all(isinstance(layer, AWQColumnParallelLinear) for layer in layers):
            awq_layers = tuple(
                layer for layer in layers if isinstance(layer, AWQColumnParallelLinear)
            )
            group_sizes = {layer.group_size for layer in awq_layers}
            if len(group_sizes) != 1:
                raise ValueError("merged AWQ layers must use the same group_size")
            qweight, qzeros, scales, bias = _share_merged_awq_storage(awq_layers)
            return self._operation(
                qweight,
                qzeros,
                scales,
                awq_layers[0].group_size,
                bias,
            )
        # modules_to_not_convert 可以让一个融合组同时包含 Dense 与 AWQ。
        # 这类少见配置保持正确性，但不会伪装成单个融合 kernel。
        return _ConcatenatedOperation(tuple(self.prepare_local(layer) for layer in layers))


def create_awq_linear_method(
    config: Mapping[str, object],
    spec: object,
) -> AWQLinearMethod:
    """Catalog 使用的顶层 factory；保持可 pickle，便于 torchrun 装配。"""

    dtype = getattr(spec, "dtype", None)
    if dtype is not torch.float16:
        raise ValueError("AWQ W4A16 first-stage runtime requires float16 activations")
    backend = getattr(spec, "quantization_backend", "auto")
    if not isinstance(backend, str):
        raise ValueError("AWQ quantization backend must be a string")
    from light_vllm.modeling.quantization.awq_schemes import select_awq_scheme

    scheme = select_awq_scheme(backend, device=getattr(spec, "device", "cpu"))
    return AWQLinearMethod(AWQConfig.from_mapping(config), scheme)
