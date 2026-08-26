"""读取 Hugging Face 兼容的本地 safetensors 模型快照。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open
from torch import Tensor, nn

from light_vllm.modeling.loaders.torch import ModelLoadError, _prepare_for_inference
from light_vllm.modeling.models.interfaces import ModelFactory, ModelSpec
from light_vllm.modeling.quantization.interfaces import QuantizationMethodFactory
from light_vllm.modeling.registry import Registry
from light_vllm.modeling.tensor_parallel import TensorShardSpec, checkpoint_shards


def _read_json(path: Path) -> Mapping[str, object]:
    """读取快照里的配置或分片索引。"""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelLoadError(f"cannot read checkpoint metadata {path}") from exc
    if not isinstance(value, dict):
        raise ModelLoadError(f"checkpoint metadata {path} must contain an object")
    return value


def _resolve_checkpoint(spec: ModelSpec) -> tuple[Path, tuple[Path, ...]]:
    """找到配置目录，并按索引顺序返回全部权重文件。"""

    if spec.weights is None:
        raise ModelLoadError("the safetensors loader requires ModelSpec.weights")
    weights = Path(spec.weights)
    if weights.is_file():
        # 单文件快照仍从同目录读取 config.json。
        if weights.suffix != ".safetensors":
            raise ModelLoadError("safetensors weights must use the .safetensors suffix")
        return weights.parent, (weights,)
    if not weights.is_dir():
        raise ModelLoadError(f"checkpoint path {weights} does not exist")

    index_path = weights / "model.safetensors.index.json"
    if index_path.is_file():
        # 大模型会拆成多个文件，索引记录每个参数在哪个分片。
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ModelLoadError("safetensors index must contain a non-empty weight_map")
        configured_shards = tuple(weight_map.values())
        if any(not isinstance(name, str) or not name for name in configured_shards):
            raise ModelLoadError("safetensors index contains an invalid shard name")
        shard_names = tuple(dict.fromkeys(configured_shards))
        root = weights.resolve()
        shards = tuple((weights / name).resolve() for name in shard_names)
        if any(path.parent != root for path in shards):
            raise ModelLoadError(
                "safetensors index shard must stay inside the checkpoint directory"
            )
    else:
        # 小模型通常没有索引，直接读取目录中的权重文件。
        shards = tuple(sorted(weights.glob("*.safetensors")))
    if not shards:
        raise ModelLoadError(f"checkpoint directory {weights} contains no safetensors weights")
    missing = tuple(path for path in shards if not path.is_file())
    if missing:
        raise ModelLoadError(f"checkpoint shards do not exist: {missing!r}")
    return weights, shards


def _quantization_config(
    checkpoint_dir: Path,
    checkpoint_args: Mapping[str, object],
) -> Mapping[str, object] | None:
    """读取 HF config 内嵌或 AutoAWQ 独立的量化配置。"""

    embedded = checkpoint_args.get("quantization_config")
    if embedded is not None:
        if not isinstance(embedded, dict):
            raise ModelLoadError("quantization_config must contain an object")
        return embedded
    for filename in ("quantize_config.json", "quant_config.json"):
        path = checkpoint_dir / filename
        if path.is_file():
            return _read_json(path)
    return None


def _resolve_linear_method(
    spec: ModelSpec,
    checkpoint_dir: Path,
    checkpoint_args: Mapping[str, object],
    quantizations: Registry[QuantizationMethodFactory] | None,
):
    """在模型构造前把 checkpoint 格式解析成 LinearMethod。"""

    if spec.linear_method is not None:
        return spec.linear_method
    config = _quantization_config(checkpoint_dir, checkpoint_args)
    configured_name = None if config is None else config.get("quant_method")
    if configured_name is not None and not isinstance(configured_name, str):
        raise ModelLoadError("quantization method must be a string")
    checkpoint_name = None if configured_name is None else configured_name.lower()
    if not isinstance(spec.quantization, str):
        raise ModelLoadError("quantization selection must be a string")
    requested = spec.quantization.lower()
    if requested == "none":
        if checkpoint_name is not None:
            raise ModelLoadError("quantization=none cannot load a quantized checkpoint")
        return None
    if requested == "auto":
        selected = checkpoint_name
    else:
        selected = requested
        if checkpoint_name is None:
            raise ModelLoadError(
                f"quantization={selected!r} requires checkpoint quantization metadata"
            )
        if checkpoint_name != selected:
            raise ModelLoadError(
                f"checkpoint quantization is {checkpoint_name!r}, not {selected!r}"
            )
    if selected is None:
        return None
    if config is None:
        raise ModelLoadError("quantized checkpoint is missing quantization configuration")
    if quantizations is None:
        raise ModelLoadError("this loader has no quantization methods registered")
    try:
        factory = quantizations.get(selected)
        return factory(config, spec)
    except (LookupError, RuntimeError, TypeError, ValueError) as exc:
        raise ModelLoadError(f"cannot configure quantization {selected!r}: {exc}") from exc


def _optional_weight_keys(model: nn.Module) -> frozenset[str]:
    """返回可以不单独保存的共享权重名称。"""

    value = getattr(model, "optional_weight_keys", frozenset())
    keys = frozenset(value)
    if any(not isinstance(key, str) or not key for key in keys):
        raise ModelLoadError("model optional_weight_keys must contain non-empty strings")
    return keys


def _copy_weight(
    name: str,
    source: Tensor,
    target: Tensor,
    shard: TensorShardSpec | None = None,
) -> None:
    """把完整权重或其中一个声明式切片复制到目标参数。"""

    if shard is not None:
        if shard.dimension >= source.ndim:
            raise ModelLoadError(f"weight {name!r} has no shard dimension {shard.dimension}")
        if source.shape[shard.dimension] != shard.full_size:
            raise ModelLoadError(
                f"weight {name!r} shard dimension has size "
                f"{source.shape[shard.dimension]}, expected {shard.full_size}"
            )
        # 与 vLLM/SGLang 的通用 weight-loader 方式一致：checkpoint 保持
        # HF 原始布局，并行层只声明本 Rank 应复制的连续区间。
        source = source.narrow(shard.dimension, shard.start, shard.length)

    if source.shape != target.shape:
        raise ModelLoadError(
            f"weight {name!r} has shape {tuple(source.shape)}, expected {tuple(target.shape)}"
        )
    try:
        target.copy_(source)
    except RuntimeError as exc:
        raise ModelLoadError(f"cannot copy checkpoint weight {name!r}") from exc


class SafetensorsModelLoader:
    """从本地 HF 兼容目录读取配置和一个或多个权重分片。"""

    def __init__(
        self,
        *,
        quantizations: Registry[QuantizationMethodFactory] | None = None,
    ) -> None:
        self._quantizations = quantizations

    def load(self, spec: ModelSpec, factory: ModelFactory) -> nn.Module:
        checkpoint_dir, shards = _resolve_checkpoint(spec)
        config_path = checkpoint_dir / "config.json"
        checkpoint_args = dict(_read_json(config_path))
        # 显式 model_args 只用于小范围覆盖；真实模型尺寸仍会由权重形状校验。
        checkpoint_args.update(spec.model_args)
        linear_method = _resolve_linear_method(
            spec,
            checkpoint_dir,
            checkpoint_args,
            self._quantizations,
        )
        resolved_spec = replace(
            spec,
            model_args=checkpoint_args,
            linear_method=linear_method,
        )
        # 先按配置创建空模型，权重文件只负责填充参数，不负责定义结构。
        model = factory(resolved_spec).to(device=spec.device, dtype=spec.dtype).eval()
        targets = model.state_dict()
        parameter_shards = checkpoint_shards(model)
        optional = _optional_weight_keys(model)
        loaded: set[str] = set()
        unexpected: set[str] = set()

        with torch.no_grad():
            for shard in shards:
                try:
                    # 一次只打开一个分片，避免把整套权重同时放进内存。
                    with safe_open(shard, framework="pt", device="cpu") as reader:
                        for name in tuple(reader.keys()):
                            if name in loaded:
                                raise ModelLoadError(
                                    f"checkpoint contains duplicate weight {name!r}"
                                )
                            loaded.add(name)
                            target = targets.get(name)
                            if target is None:
                                unexpected.add(name)
                                continue
                            _copy_weight(
                                name,
                                reader.get_tensor(name),
                                target,
                                parameter_shards.get(name),
                            )
                except (OSError, SafetensorError) as exc:
                    raise ModelLoadError(f"cannot read checkpoint shard {shard}") from exc

        missing = set(targets).difference(loaded, optional)
        # 全部分片读完再统一报错，能一次看清缺失和多余的权重。
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"missing={sorted(missing)!r}")
            if unexpected:
                details.append(f"unexpected={sorted(unexpected)!r}")
            raise ModelLoadError("checkpoint weights do not match the model: " + ", ".join(details))
        return _prepare_for_inference(model, spec)
