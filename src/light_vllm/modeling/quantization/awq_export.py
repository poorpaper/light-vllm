"""把完成 scale/clip 搜索的 Dense 模型导出为 AutoAWQ checkpoint。"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path

from safetensors.torch import save_file
from torch import Tensor, nn

from light_vllm.modeling.quantization.awq import pack_awq_checkpoint_weight
from light_vllm.modeling.quantization.awq_ptq import AWQPTQConfig, LayerPTQReport

_QWEN2_ASSET_NAMES = (
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)


@contextmanager
def atomic_output_directory(output_dir: str | Path) -> Iterator[Path]:
    """在同一文件系统内准备完整目录，成功后一次发布。

    safetensors、配置或 tokenizer 任一步失败时，只删除本次创建的 staging
    目录；调用者指定的目标和已有文件永远不会被覆盖。
    """

    root = Path(output_dir)
    if root.exists():
        raise FileExistsError(f"AWQ output path {root} already exists")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent))
    try:
        yield staging
        staging.replace(root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _tensor_bytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _safe_cpu_tensor(tensor: Tensor) -> Tensor:
    # safetensors 拒绝共享存储。量化 Linear 的 weight 被替换，tied lm_head
    # 另行省略；其余张量 clone 一次，保证每个导出项拥有独立连续存储。
    return tensor.detach().cpu().contiguous().clone()


def copy_qwen2_checkpoint_assets(source_dir: str | Path, output_dir: str | Path) -> tuple[str, ...]:
    """原样复制不属于权重的 Qwen tokenizer 与生成配置。

    tokenizer 的序列化并非恒等操作，跨 Transformers 版本重新保存可能改写
    regex 或 chat template。PTQ 只改变权重，因此这些文件必须保持字节一致。
    """

    source = Path(source_dir)
    output = Path(output_dir)
    if not source.is_dir() or not output.is_dir():
        raise FileNotFoundError("AWQ asset source and output must both be directories")
    copied: list[str] = []
    for name in _QWEN2_ASSET_NAMES:
        asset = source / name
        if not asset.is_file():
            continue
        destination = output / name
        if destination.exists():
            raise FileExistsError(f"AWQ output asset {destination} already exists")
        shutil.copy2(asset, destination)
        copied.append(name)
    if not any(name.startswith("tokenizer") for name in copied):
        raise FileNotFoundError("Qwen2 checkpoint contains no tokenizer assets")
    return tuple(copied)


def _iter_awq_state(
    model: nn.Module,
    quantized_modules: Sequence[str],
    *,
    group_size: int,
) -> Iterator[tuple[str, Tensor]]:
    state = model.state_dict()
    replaced_weights = {f"{name}.weight" for name in quantized_modules}
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    for name, tensor in state.items():
        if name in replaced_weights:
            packed = pack_awq_checkpoint_weight(tensor, group_size)
            prefix = name.removesuffix(".weight")
            yield f"{prefix}.qweight", _safe_cpu_tensor(packed.qweight)
            yield f"{prefix}.qzeros", _safe_cpu_tensor(packed.qzeros)
            yield f"{prefix}.scales", _safe_cpu_tensor(packed.scales)
        elif tied and name == "lm_head.weight":
            # light-vllm 和 HF 都能根据 tie_word_embeddings 恢复共享权重。
            continue
        else:
            yield name, _safe_cpu_tensor(tensor)


def _write_shards(
    output_dir: Path,
    tensors: Iterator[tuple[str, Tensor]],
    *,
    max_shard_bytes: int,
) -> tuple[dict[str, str], int]:
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    temporary: list[tuple[Path, tuple[str, ...]]] = []
    shard: dict[str, Tensor] = {}
    shard_size = 0
    total_size = 0

    def flush() -> None:
        nonlocal shard, shard_size
        if not shard:
            return
        path = output_dir / f".awq-part-{len(temporary) + 1:05d}.safetensors"
        save_file(shard, path, metadata={"format": "pt"})
        temporary.append((path, tuple(shard)))
        shard = {}
        shard_size = 0

    for name, tensor in tensors:
        size = _tensor_bytes(tensor)
        if shard and shard_size + size > max_shard_bytes:
            flush()
        shard[name] = tensor
        shard_size += size
        total_size += size
    flush()
    if not temporary:
        raise ValueError("cannot export an empty model state")

    weight_map: dict[str, str] = {}
    total_shards = len(temporary)
    for index, (temporary_path, names) in enumerate(temporary, start=1):
        filename = f"model-{index:05d}-of-{total_shards:05d}.safetensors"
        final_path = output_dir / filename
        temporary_path.replace(final_path)
        weight_map.update({name: filename for name in names})
    return weight_map, total_size


def export_awq_checkpoint(
    model: nn.Module,
    output_dir: str | Path,
    quantized_modules: Sequence[str],
    *,
    group_size: int,
    reports: Sequence[LayerPTQReport] = (),
    ptq_config: AWQPTQConfig | None = None,
    calibration: Mapping[str, object] | None = None,
    max_shard_bytes: int = 2 * 1024**3,
) -> Path:
    """把量化模型写入一个空目录；目录级原子性由外层上下文负责。"""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"AWQ export directory {root} must be empty")

    weight_map, total_size = _write_shards(
        root,
        _iter_awq_state(model, quantized_modules, group_size=group_size),
        max_shard_bytes=max_shard_bytes,
    )
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    quantization_config: dict[str, object] = {
        "quant_method": "awq",
        "bits": 4,
        "group_size": group_size,
        "zero_point": True,
        "version": "GEMM",
    }
    skipped = tuple(
        name
        for name in getattr(model, "_light_vllm_modules_to_not_convert", ())
        if isinstance(name, str)
    )
    if skipped:
        quantization_config["modules_to_not_convert"] = list(skipped)
    if hasattr(model.config, "to_dict"):
        config = model.config.to_dict()
    elif is_dataclass(model.config):
        config = asdict(model.config)
    else:
        raise TypeError("exported model config must provide to_dict or be a dataclass")
    config["quantization_config"] = quantization_config
    (root / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    autoawq_config = {
        "quant_method": "awq",
        "w_bit": 4,
        "q_group_size": group_size,
        "zero_point": True,
        "version": "GEMM",
    }
    if skipped:
        autoawq_config["modules_to_not_convert"] = list(skipped)
    (root / "quantize_config.json").write_text(
        json.dumps(autoawq_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "format": "awq-w4a16-autoawq-gemm",
        "group_size": group_size,
        "quantized_modules": list(quantized_modules),
        "layers": [asdict(value) for value in reports],
    }
    if ptq_config is not None:
        report["ptq_config"] = asdict(ptq_config)
    if calibration is not None:
        report["calibration"] = dict(calibration)
    (root / "awq_ptq_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return root
