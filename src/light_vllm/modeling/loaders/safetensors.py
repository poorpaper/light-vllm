"""读取 Hugging Face 兼容的本地 safetensors 模型快照。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open
from torch import Tensor, nn

from light_vllm.modeling.loaders.torch import ModelLoadError
from light_vllm.modeling.models.interfaces import ModelFactory, ModelSpec


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelLoadError(f"cannot read checkpoint metadata {path}") from exc
    if not isinstance(value, dict):
        raise ModelLoadError(f"checkpoint metadata {path} must contain an object")
    return value


def _resolve_checkpoint(spec: ModelSpec) -> tuple[Path, tuple[Path, ...]]:
    if spec.weights is None:
        raise ModelLoadError("the safetensors loader requires ModelSpec.weights")
    weights = Path(spec.weights)
    if weights.is_file():
        if weights.suffix != ".safetensors":
            raise ModelLoadError("safetensors weights must use the .safetensors suffix")
        return weights.parent, (weights,)
    if not weights.is_dir():
        raise ModelLoadError(f"checkpoint path {weights} does not exist")

    index_path = weights / "model.safetensors.index.json"
    if index_path.is_file():
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
        shards = tuple(sorted(weights.glob("*.safetensors")))
    if not shards:
        raise ModelLoadError(f"checkpoint directory {weights} contains no safetensors weights")
    missing = tuple(path for path in shards if not path.is_file())
    if missing:
        raise ModelLoadError(f"checkpoint shards do not exist: {missing!r}")
    return weights, shards


def _optional_weight_keys(model: nn.Module) -> frozenset[str]:
    value = getattr(model, "optional_weight_keys", frozenset())
    keys = frozenset(value)
    if any(not isinstance(key, str) or not key for key in keys):
        raise ModelLoadError("model optional_weight_keys must contain non-empty strings")
    return keys


def _copy_weight(name: str, source: Tensor, target: Tensor) -> None:
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

    def load(self, spec: ModelSpec, factory: ModelFactory) -> nn.Module:
        checkpoint_dir, shards = _resolve_checkpoint(spec)
        config_path = checkpoint_dir / "config.json"
        checkpoint_args = dict(_read_json(config_path))
        # 显式 model_args 只用于小范围覆盖；真实模型尺寸仍会由权重形状校验。
        checkpoint_args.update(spec.model_args)
        resolved_spec = replace(spec, model_args=checkpoint_args)
        model = factory(resolved_spec).to(device=spec.device, dtype=spec.dtype).eval()
        targets = model.state_dict()
        optional = _optional_weight_keys(model)
        loaded: set[str] = set()
        unexpected: set[str] = set()

        with torch.no_grad():
            for shard in shards:
                try:
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
                            _copy_weight(name, reader.get_tensor(name), target)
                except (OSError, SafetensorError) as exc:
                    raise ModelLoadError(f"cannot read checkpoint shard {shard}") from exc

        missing = set(targets).difference(loaded, optional)
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"missing={sorted(missing)!r}")
            if unexpected:
                details.append(f"unexpected={sorted(unexpected)!r}")
            raise ModelLoadError("checkpoint weights do not match the model: " + ", ".join(details))
        return model
