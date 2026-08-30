"""AWQ PTQ 的激活感知缩放、裁剪搜索与 Qwen2 层级编排。

这里实现离线量化算法，不进入在线推理依赖图。运行时只消费最终的
``qweight/qzeros/scales``，不会携带 calibration 样本或搜索状态。
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from light_vllm.modeling.quantization.awq import _quantize_awq_groups


@dataclass(frozen=True, slots=True)
class AWQPTQConfig:
    """离线 AWQ 搜索预算。"""

    group_size: int = 128
    num_scale_steps: int = 20
    num_clip_steps: int = 20
    max_clip_shrink: float = 0.5
    max_clip_tokens: int = 512
    max_parallel_calibration_samples: int = 8
    modules_to_not_convert: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("AWQ PTQ group_size must be positive or -1")
        if self.num_scale_steps <= 0 or self.num_clip_steps <= 0:
            raise ValueError("AWQ PTQ search steps must be positive")
        if not 0 < self.max_clip_shrink <= 1:
            raise ValueError("AWQ max_clip_shrink must be within (0, 1]")
        if self.max_clip_tokens <= 0:
            raise ValueError("AWQ max_clip_tokens must be positive")
        if self.max_parallel_calibration_samples <= 0:
            raise ValueError("AWQ parallel calibration sample count must be positive")
        if any(
            not isinstance(name, str) or not name.strip() for name in self.modules_to_not_convert
        ):
            raise ValueError("AWQ skipped module names must be non-empty strings")


@dataclass(frozen=True, slots=True)
class ScaleSearchResult:
    scales: Tensor
    ratio: float
    error: float


@dataclass(frozen=True, slots=True)
class ClipSearchResult:
    max_values: Tensor
    mean_error: float


@dataclass(frozen=True, slots=True)
class LayerPTQReport:
    layer_index: int
    scale_ratios: Mapping[str, float]
    scale_errors: Mapping[str, float]
    clip_errors: Mapping[str, float]


def pseudo_quantize_weight(weight: Tensor, group_size: int) -> Tensor:
    """模拟非对称 INT4 量化，返回反量化后的同形状权重。"""

    if weight.ndim < 2 or not weight.is_floating_point():
        raise ValueError("AWQ pseudo quantization requires floating weights")
    quantized, zeros, scales = _quantize_awq_groups(weight, group_size)
    return ((quantized - zeros).to(scales.dtype) * scales).reshape_as(weight)


def _channel_weight_importance(linears: Sequence[nn.Linear], group_size: int) -> Tensor:
    width = linears[0].in_features
    effective_group_size = width if group_size == -1 else group_size
    if width % effective_group_size:
        raise ValueError("AWQ group_size must divide every searched Linear input")
    total = torch.zeros(width, dtype=torch.float32)
    total_rows = 0
    # 不拼接 Q/K/V 或 Gate/Up 的完整权重；按约 64 MiB 的 FP32 工作集归并
    # 通道统计，7B/14B 校准时不会额外复制一份巨大的融合矩阵。
    rows_per_chunk = max(1, (64 * 1024**2) // (width * 4))
    for layer in linears:
        if layer.in_features != width:
            raise ValueError("AWQ searched linears must have the same input width")
        weight = layer.weight.detach()
        for chunk in weight.split(rows_per_chunk, dim=0):
            grouped = chunk.float().reshape(-1, effective_group_size)
            absolute = grouped.abs()
            normalized = absolute / (absolute.amax(dim=1, keepdim=True) + 1e-6)
            total.add_(normalized.reshape(chunk.shape[0], width).sum(dim=0).cpu())
        total_rows += weight.shape[0]
    return total / total_rows


def _channel_activation_importance(inputs: Tensor) -> Tensor:
    flattened = inputs.detach().reshape(-1, inputs.shape[-1])
    if flattened.shape[0] == 0:
        raise ValueError("AWQ calibration inputs must contain tokens")
    width = flattened.shape[1]
    total = torch.zeros(width, dtype=torch.float32)
    rows_per_chunk = max(1, (64 * 1024**2) // (width * 4))
    for chunk in flattened.split(rows_per_chunk, dim=0):
        total.add_(chunk.float().abs().sum(dim=0).cpu())
    return total / flattened.shape[0]


@torch.no_grad()
def search_awq_scale(
    inputs: Tensor,
    linears: Sequence[nn.Linear],
    *,
    group_size: int,
    num_steps: int = 20,
    inspect_module: nn.Module | None = None,
    module_kwargs: Mapping[str, object] | None = None,
    max_parallel_samples: int = 8,
) -> ScaleSearchResult:
    """搜索 AWQ 通道缩放，最小化目标模块的真实输出误差。

    目标式与 AWQ 相同：``Q(W*s)/s * X`` 对齐 ``W*X``。搜索阶段把
    ``1/s`` 折回反量化权重，因此不修改模型；选定后再把 ``s`` 等价地
    分摊到前驱算子和目标 Linear。Q/K/V 和 Gate/Up 不能只比较各自的
    Linear 输出：它们还要经过 attention 或 SiLU 组合，因此这里和
    AutoAWQ 一样允许用完整子模块计算 loss。
    """

    if not linears:
        raise ValueError("AWQ scale search requires at least one Linear")
    width = inputs.shape[-1]
    if any(layer.in_features != width for layer in linears):
        raise ValueError("AWQ scale search inputs must match every Linear")
    if num_steps <= 0:
        raise ValueError("AWQ scale search steps must be positive")
    if max_parallel_samples <= 0:
        raise ValueError("AWQ parallel calibration sample count must be positive")
    if inspect_module is None:
        if len(linears) != 1:
            raise ValueError("multiple AWQ linears require an inspect_module")
        inspect_module = linears[0]
    kwargs = _sanitize_kwargs(module_kwargs or {}, inspect_module)
    target_device = linears[0].weight.device
    x_mean = _channel_activation_importance(inputs).to(target_device)
    w_mean = _channel_weight_importance(linears, group_size).to(target_device)
    batch_size = inputs.shape[0]
    reference_chunks: list[Tensor] = []
    for start in range(0, batch_size, max_parallel_samples):
        end = min(start + max_parallel_samples, batch_size)
        chunk_inputs = inputs[start:end].to(target_device)
        chunk_kwargs = _batch_kwargs(kwargs, start, end, batch_size, target_device)
        reference_chunks.append(
            _module_output(inspect_module(chunk_inputs, **chunk_kwargs)).detach().cpu()
        )
    # 只备份本次会被替换的权重。比复制整个 DecoderLayer 更省内存，也避免
    # state_dict 在 CPU 模型上返回共享 storage 而导致“备份”被一起改写。
    original_weights = tuple(layer.weight.detach().clone() for layer in linears)
    best_scales: Tensor | None = None
    best_ratio = -1.0
    best_error = float("inf")
    try:
        for index in range(num_steps):
            ratio = index / num_steps
            scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp_min(1e-4)
            scales = scales / torch.sqrt(scales.max() * scales.min())
            scales = torch.where(torch.isfinite(scales), scales, torch.ones_like(scales))
            for layer, original in zip(linears, original_weights, strict=True):
                typed_scales = scales.to(layer.weight.dtype).view(1, -1)
                candidate = pseudo_quantize_weight(original * typed_scales, group_size)
                layer.weight.copy_(candidate / typed_scales)
            squared_error = 0.0
            value_count = 0
            for chunk_index, start in enumerate(range(0, batch_size, max_parallel_samples)):
                end = min(start + max_parallel_samples, batch_size)
                chunk_inputs = inputs[start:end].to(target_device)
                chunk_kwargs = _batch_kwargs(kwargs, start, end, batch_size, target_device)
                candidate_output = _module_output(inspect_module(chunk_inputs, **chunk_kwargs))
                reference = reference_chunks[chunk_index].to(
                    candidate_output.device, candidate_output.dtype
                )
                difference = candidate_output.float() - reference.float()
                squared_error += difference.pow(2).sum().item()
                value_count += difference.numel()
            error = squared_error / value_count
            if error < best_error:
                best_error = error
                best_ratio = ratio
                best_scales = scales.detach().cpu()
            for layer, original in zip(linears, original_weights, strict=True):
                layer.weight.copy_(original)
    finally:
        for layer, original in zip(linears, original_weights, strict=True):
            layer.weight.copy_(original)
    if best_scales is None:
        raise RuntimeError("AWQ scale search did not evaluate any candidate")
    return ScaleSearchResult(best_scales, best_ratio, best_error)


@torch.no_grad()
def apply_awq_scale(
    previous: nn.Module,
    linears: Sequence[nn.Linear],
    scales: Tensor,
) -> None:
    """把选中的 scale 等价折叠进前驱输出与后继 Linear 输入列。"""

    if not linears:
        raise ValueError("AWQ scale application requires target linears")
    device_scales = scales.to(linears[0].weight.device, linears[0].weight.dtype)
    if any(layer.in_features != device_scales.numel() for layer in linears):
        raise ValueError("AWQ scales must match target Linear input width")
    if isinstance(previous, nn.Linear):
        if previous.out_features != device_scales.numel():
            raise ValueError("AWQ scales must match previous Linear output width")
        previous.weight.div_(device_scales.view(-1, 1))
        if previous.bias is not None:
            previous.bias.div_(device_scales)
    elif hasattr(previous, "weight"):
        weight = previous.weight
        if not isinstance(weight, Tensor) or weight.ndim != 1:
            raise TypeError("AWQ norm predecessor must expose a one-dimensional weight")
        if weight.numel() != device_scales.numel():
            raise ValueError("AWQ scales must match predecessor norm width")
        weight.div_(device_scales)
    else:
        raise TypeError("AWQ scaling supports Linear or normalization predecessors")
    for layer in linears:
        layer.weight.mul_(device_scales.view(1, -1))


@torch.no_grad()
def search_awq_clip(
    weight: Tensor,
    inputs: Tensor,
    *,
    group_size: int,
    num_steps: int = 20,
    max_shrink: float = 0.5,
    max_tokens: int = 512,
    output_chunk_size: int = 64,
) -> ClipSearchResult:
    """逐输出通道、逐输入组搜索对称裁剪上界。

    裁剪损失在每个 group 的部分点积上计算，输出通道分块处理，避免为 7B
    模型一次物化 ``out_channels × tokens × groups`` 的巨大临时张量。
    """

    if weight.ndim != 2 or inputs.shape[-1] != weight.shape[1]:
        raise ValueError("AWQ clipping inputs must match a two-dimensional weight")
    effective_group_size = weight.shape[1] if group_size == -1 else group_size
    if weight.shape[1] % effective_group_size:
        raise ValueError("AWQ group_size must divide clipped Linear input width")
    flattened = inputs.detach().reshape(-1, inputs.shape[-1])
    step = max(1, flattened.shape[0] // max_tokens)
    sampled = flattened[::step][:max_tokens]
    sampled = sampled.reshape(1, sampled.shape[0], -1, effective_group_size)
    num_candidates = max(1, int(num_steps * max_shrink))
    best_chunks = []
    total_error = 0.0
    total_values = 0
    for start in range(0, weight.shape[0], output_chunk_size):
        chunk = weight[start : start + output_chunk_size]
        grouped = chunk.reshape(chunk.shape[0], 1, -1, effective_group_size)
        chunk_inputs = sampled.to(grouped.device, grouped.dtype)
        reference = (chunk_inputs * grouped).sum(dim=-1)
        original_max = grouped.abs().amax(dim=-1, keepdim=True)
        best_max = original_max.clone()
        best_error = torch.full(
            original_max.shape,
            torch.inf,
            dtype=torch.float32,
            device=original_max.device,
        )
        for index in range(num_candidates):
            candidate_max = original_max * (1 - index / num_steps)
            clipped = torch.clamp(grouped, -candidate_max, candidate_max)
            quantized = pseudo_quantize_weight(clipped, effective_group_size)
            candidate = (chunk_inputs * quantized).sum(dim=-1)
            error = (candidate - reference).float().pow(2).mean(dim=1).view_as(best_error)
            improved = error < best_error
            best_error[improved] = error[improved]
            best_max[improved] = candidate_max[improved]
        best_chunks.append(best_max.squeeze(1).cpu())
        total_error += best_error.sum().item()
        total_values += best_error.numel()
    return ClipSearchResult(torch.cat(best_chunks, dim=0), total_error / total_values)


@torch.no_grad()
def apply_awq_clip(weight: Tensor, max_values: Tensor, group_size: int) -> None:
    effective_group_size = weight.shape[1] if group_size == -1 else group_size
    grouped = weight.reshape(weight.shape[0], -1, effective_group_size)
    expected = (weight.shape[0], grouped.shape[1], 1)
    if max_values.shape != expected:
        raise ValueError(
            f"AWQ clip values have shape {tuple(max_values.shape)}, expected {expected}"
        )
    limits = max_values.to(weight.device, weight.dtype)
    grouped.copy_(torch.clamp(grouped, -limits, limits))


def _sanitize_kwargs(kwargs: Mapping[str, object], module: nn.Module) -> dict[str, object]:
    parameters = inspect.signature(module.forward).parameters
    return {
        name: value
        for name, value in kwargs.items()
        if name != "hidden_states" and name in parameters
    }


def _move_value(value: object, device: torch.device) -> object:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_value(item, device) for item in value)
    if isinstance(value, list):
        return [_move_value(item, device) for item in value]
    if isinstance(value, Mapping):
        return {name: _move_value(item, device) for name, item in value.items()}
    return value


def _slice_batch_value(
    value: object,
    start: int,
    end: int,
    batch_size: int,
) -> object:
    """只切首维确实属于 calibration batch 的嵌套参数。"""

    if isinstance(value, Tensor):
        if value.ndim and value.shape[0] == batch_size:
            return value[start:end]
        return value
    if isinstance(value, tuple):
        return tuple(_slice_batch_value(item, start, end, batch_size) for item in value)
    if isinstance(value, list):
        return [_slice_batch_value(item, start, end, batch_size) for item in value]
    if isinstance(value, Mapping):
        return {
            name: _slice_batch_value(item, start, end, batch_size) for name, item in value.items()
        }
    return value


def _batch_kwargs(
    kwargs: Mapping[str, object],
    start: int,
    end: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    return {
        name: _move_value(
            _slice_batch_value(value, start, end, batch_size),
            device,
        )
        for name, value in kwargs.items()
    }


def _merge_batch_values(values: Sequence[object], batch_sizes: Sequence[int]) -> object:
    """合并多次 first-layer 捕获值；广播常量必须逐批一致。"""

    if not values or len(values) != len(batch_sizes):
        raise ValueError("AWQ calibration captures and batch sizes must align")
    first = values[0]
    if isinstance(first, Tensor):
        tensors = tuple(value for value in values if isinstance(value, Tensor))
        if len(tensors) != len(values):
            raise TypeError("AWQ calibration captured inconsistent tensor values")
        if all(
            value.ndim and value.shape[0] == size
            for value, size in zip(tensors, batch_sizes, strict=True)
        ):
            return torch.cat(tuple(value.cpu() for value in tensors), dim=0)
        reference = first.cpu()
        if any(
            value.shape != first.shape or not torch.equal(value.cpu(), reference)
            for value in tensors[1:]
        ):
            raise ValueError("AWQ calibration captured a non-batched value that changes by batch")
        return reference
    if isinstance(first, tuple):
        if any(not isinstance(value, tuple) or len(value) != len(first) for value in values):
            raise TypeError("AWQ calibration captured inconsistent tuple values")
        return tuple(
            _merge_batch_values(
                tuple(value[index] for value in values),  # type: ignore[index]
                batch_sizes,
            )
            for index in range(len(first))
        )
    if isinstance(first, list):
        if any(not isinstance(value, list) or len(value) != len(first) for value in values):
            raise TypeError("AWQ calibration captured inconsistent list values")
        return [
            _merge_batch_values(
                tuple(value[index] for value in values),  # type: ignore[index]
                batch_sizes,
            )
            for index in range(len(first))
        ]
    if isinstance(first, Mapping):
        if any(not isinstance(value, Mapping) or value.keys() != first.keys() for value in values):
            raise TypeError("AWQ calibration captured inconsistent mapping values")
        return {
            name: _merge_batch_values(
                tuple(value[name] for value in values),  # type: ignore[index]
                batch_sizes,
            )
            for name in first
        }
    if any(value != first for value in values[1:]):
        raise ValueError("AWQ calibration captured a constant that changes by batch")
    return first


def _module_output(value: object) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], Tensor):
        return value[0]
    raise TypeError("Qwen2 decoder layer must return hidden states as its first value")


def _qwen2_linears(layer: nn.Module) -> dict[str, nn.Linear]:
    paths = {
        "self_attn.q_proj": layer.self_attn.q_proj,
        "self_attn.k_proj": layer.self_attn.k_proj,
        "self_attn.v_proj": layer.self_attn.v_proj,
        "self_attn.o_proj": layer.self_attn.o_proj,
        "mlp.gate_proj": layer.mlp.gate_proj,
        "mlp.up_proj": layer.mlp.up_proj,
        "mlp.down_proj": layer.mlp.down_proj,
    }
    if any(not isinstance(module, nn.Linear) for module in paths.values()):
        raise TypeError("AWQ PTQ expects an unquantized Qwen2 model with nn.Linear projections")
    return paths


def _capture_linear_inputs(
    layer: nn.Module,
    hidden_states: Tensor,
    layer_kwargs: Mapping[str, object],
    *,
    max_parallel_samples: int,
) -> tuple[Tensor, dict[str, Tensor]]:
    linears = _qwen2_linears(layer)
    captured_chunks: dict[str, list[Tensor]] = {}
    handles = []

    def capture(name: str):
        def hook(_module: nn.Module, inputs: tuple[object, ...], _output: object) -> None:
            value = inputs[0]
            if not isinstance(value, Tensor):
                raise TypeError("AWQ calibration hook expected a Tensor input")
            captured_chunks.setdefault(name, []).append(value.detach().cpu())

        return hook

    # Q/K/V 共享输入，Gate/Up 也共享；只保存四份独立激活，避免校准内存翻倍。
    for name in ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj"):
        handles.append(linears[name].register_forward_hook(capture(name)))
    try:
        outputs: list[Tensor] = []
        batch_size = hidden_states.shape[0]
        kwargs = _sanitize_kwargs(layer_kwargs, layer)
        device = next(layer.parameters()).device
        for start in range(0, batch_size, max_parallel_samples):
            end = min(start + max_parallel_samples, batch_size)
            chunk_kwargs = _batch_kwargs(kwargs, start, end, batch_size, device)
            output = layer(hidden_states[start:end].to(device), **chunk_kwargs)
            outputs.append(_module_output(output).detach().cpu())
    finally:
        for handle in handles:
            handle.remove()
    next_hidden = torch.cat(outputs, dim=0)
    captured = {name: torch.cat(chunks, dim=0) for name, chunks in captured_chunks.items()}
    captured["self_attn.k_proj"] = captured["self_attn.q_proj"]
    captured["self_attn.v_proj"] = captured["self_attn.q_proj"]
    captured["mlp.up_proj"] = captured["mlp.gate_proj"]
    return next_hidden, captured


@torch.no_grad()
def quantize_qwen2_layers(
    model: nn.Module,
    hidden_states: Tensor,
    layer_kwargs: Mapping[str, object],
    config: AWQPTQConfig,
    *,
    device: str | torch.device,
) -> tuple[tuple[str, ...], tuple[LayerPTQReport, ...]]:
    """逐层搜索 Qwen2 权重，返回需要打包导出的 Linear 全名。"""

    layers = model.model.layers
    target_device = torch.device(device)
    quantized_names: list[str] = []
    reports: list[LayerPTQReport] = []
    current = hidden_states
    for layer_index, layer in enumerate(layers):
        layer.to(target_device)
        next_hidden, features = _capture_linear_inputs(
            layer,
            current,
            layer_kwargs,
            max_parallel_samples=config.max_parallel_calibration_samples,
        )
        linears = _qwen2_linears(layer)
        scale_groups: list[tuple[str, nn.Module, tuple[str, ...], str, nn.Module | None]] = [
            (
                "attention_input",
                layer.input_layernorm,
                ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
                "self_attn.q_proj",
                layer.self_attn,
            ),
            (
                "mlp_input",
                layer.post_attention_layernorm,
                ("mlp.gate_proj", "mlp.up_proj"),
                "mlp.gate_proj",
                layer.mlp,
            ),
            (
                "mlp_output",
                layer.mlp.up_proj,
                ("mlp.down_proj",),
                "mlp.down_proj",
                None,
            ),
        ]
        if layer.self_attn.v_proj.weight.shape == layer.self_attn.o_proj.weight.shape:
            scale_groups.insert(
                1,
                (
                    "attention_output",
                    layer.self_attn.v_proj,
                    ("self_attn.o_proj",),
                    "self_attn.o_proj",
                    None,
                ),
            )
        scale_ratios: dict[str, float] = {}
        scale_errors: dict[str, float] = {}
        for group_name, previous, target_names, input_name, inspect_module in scale_groups:
            search_names = tuple(
                name
                for name in target_names
                if not any(
                    skipped in f"model.layers.{layer_index}.{name}"
                    for skipped in config.modules_to_not_convert
                )
            )
            if not search_names:
                continue
            search_targets = tuple(linears[name] for name in search_names)
            apply_targets = tuple(linears[name] for name in target_names)
            result = search_awq_scale(
                features[input_name],
                search_targets,
                group_size=config.group_size,
                num_steps=config.num_scale_steps,
                inspect_module=inspect_module,
                module_kwargs=layer_kwargs,
                max_parallel_samples=config.max_parallel_calibration_samples,
            )
            # 前驱输出被除以 scale 后，同一输出的所有消费者都必须乘回 scale；
            # 被排除量化的兄弟 Linear 也参与等价变换，但不会被搜索或打包。
            apply_awq_scale(previous, apply_targets, result.scales)
            # 同组 Linear 可能共用同一份捕获 tensor，但字典中的多个 key 不会
            # 随一次重新赋值同步变化。逐个更新，保证后续 V/Up 的 clip 搜索看到
            # scale 已经折进前驱后的真实输入。
            for target_name in target_names:
                feature_scale = result.scales.to(features[target_name].dtype)
                features[target_name] = features[target_name] / feature_scale
            scale_ratios[group_name] = result.ratio
            scale_errors[group_name] = result.error

        clip_errors: dict[str, float] = {}
        for local_name, linear in linears.items():
            full_name = f"model.layers.{layer_index}.{local_name}"
            if any(skipped in full_name for skipped in config.modules_to_not_convert):
                continue
            # Q/K 随后的点积会放大独立裁剪误差；与 AutoAWQ 一样保留它们的范围。
            if local_name not in ("self_attn.q_proj", "self_attn.k_proj"):
                clip = search_awq_clip(
                    linear.weight,
                    features[local_name],
                    group_size=config.group_size,
                    num_steps=config.num_clip_steps,
                    max_shrink=config.max_clip_shrink,
                    max_tokens=config.max_clip_tokens,
                )
                apply_awq_clip(linear.weight, clip.max_values, config.group_size)
                clip_errors[full_name] = clip.mean_error
            quantized_names.append(full_name)
        reports.append(
            LayerPTQReport(
                layer_index=layer_index,
                scale_ratios=scale_ratios,
                scale_errors=scale_errors,
                clip_errors=clip_errors,
            )
        )
        # 下一层使用本层量化前的输出，这是 AWQ 的逐层校准定义；scale 变换在
        # Dense 数学上等价，clip/INT4 误差由各层自己的激活目标控制。
        current = next_hidden.cpu()
        layer.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return tuple(quantized_names), tuple(reports)


class _CalibrationCaptured(RuntimeError):
    pass


class _FirstLayerCatcher(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer
        self.hidden_states: Tensor | None = None
        self.kwargs: dict[str, object] = {}

    @property
    def attention_type(self) -> object:
        # Transformers 4.57 在调用 layer.forward 前先读取这一层级属性。
        return self.layer.attention_type

    def forward(self, *args: object, **kwargs: object) -> Any:
        hidden_states = kwargs.get("hidden_states")
        if isinstance(hidden_states, Tensor):
            self.hidden_states = hidden_states.detach()
        elif args and isinstance(args[0], Tensor):
            self.hidden_states = args[0].detach()
        self.kwargs = {name: value for name, value in kwargs.items() if name != "hidden_states"}
        raise _CalibrationCaptured


@torch.no_grad()
def capture_qwen2_first_layer_inputs(
    model: nn.Module,
    input_ids: Tensor,
    *,
    device: str | torch.device,
    max_parallel_samples: int = 8,
) -> tuple[Tensor, Mapping[str, object]]:
    """只执行 embedding/position 准备并截获第一层输入，避免整模上 GPU。"""

    target_device = torch.device(device)
    layers = model.model.layers
    if len(layers) == 0:
        raise ValueError("AWQ PTQ requires a model with at least one decoder layer")
    if input_ids.ndim != 2 or input_ids.shape[0] == 0:
        raise ValueError("AWQ calibration input_ids must have a non-empty batch dimension")
    if max_parallel_samples <= 0:
        raise ValueError("AWQ parallel calibration sample count must be positive")
    original = layers[0]
    catcher = _FirstLayerCatcher(original).to(target_device)
    layers[0] = catcher
    model.model.embed_tokens.to(target_device)
    rotary = getattr(model.model, "rotary_emb", None)
    if isinstance(rotary, nn.Module):
        rotary.to(target_device)
    hidden_chunks: list[Tensor] = []
    kwargs_chunks: list[Mapping[str, object]] = []
    batch_sizes: list[int] = []
    try:
        for chunk in input_ids.split(max_parallel_samples, dim=0):
            catcher.hidden_states = None
            catcher.kwargs = {}
            with suppress(_CalibrationCaptured):
                model(input_ids=chunk.to(target_device), use_cache=False)
            if catcher.hidden_states is None:
                raise RuntimeError("failed to capture Qwen2 first-layer calibration inputs")
            hidden_chunks.append(catcher.hidden_states.cpu())
            kwargs_chunks.append(catcher.kwargs)
            batch_sizes.append(chunk.shape[0])
    finally:
        layers[0] = catcher.layer
        layers[0].cpu()
        model.model.embed_tokens.cpu()
        if isinstance(rotary, nn.Module):
            rotary.cpu()
    merged_kwargs = _merge_batch_values(kwargs_chunks, batch_sizes)
    if not isinstance(merged_kwargs, Mapping):
        raise TypeError("AWQ first-layer keyword capture must be a mapping")
    return torch.cat(hidden_chunks, dim=0), merged_kwargs
