"""从 Dense Hugging Face Qwen2/Qwen2.5 checkpoint 生成 AWQ W4A16。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from light_vllm.modeling.quantization.awq_export import (
    atomic_output_directory,
    copy_qwen2_checkpoint_assets,
    export_awq_checkpoint,
)
from light_vllm.modeling.quantization.awq_ptq import (
    AWQPTQConfig,
    capture_qwen2_first_layer_inputs,
    quantize_qwen2_layers,
)
from light_vllm.modeling.quantization.calibration import (
    build_calibration_input_ids,
    iter_calibration_texts,
)

_DTYPES = {"float16": torch.float16}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantize Qwen2/Qwen2.5 to AWQ W4A16")
    parser.add_argument("--model", type=Path, required=True, help="local Dense HF checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--calibration-text-field", default="text")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="float16")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-calibration-samples", type=int, default=128)
    parser.add_argument("--calibration-sequence-length", type=int, default=512)
    parser.add_argument("--scale-search-steps", type=int, default=20)
    parser.add_argument("--clip-search-steps", type=int, default=20)
    parser.add_argument("--max-clip-shrink", type=float, default=0.5)
    parser.add_argument("--max-clip-tokens", type=int, default=512)
    parser.add_argument("--max-parallel-calibration-samples", type=int, default=8)
    parser.add_argument("--max-shard-size-gib", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--modules-to-not-convert",
        action="append",
        default=[],
        help="repeatable module-name substring kept in Dense form",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for AWQ calibration")
    if args.max_shard_size_gib <= 0:
        raise ValueError("--max-shard-size-gib must be positive")
    if args.output.exists():
        raise FileExistsError(f"AWQ output path {args.output} already exists")
    config = AWQPTQConfig(
        group_size=args.group_size,
        num_scale_steps=args.scale_search_steps,
        num_clip_steps=args.clip_search_steps,
        max_clip_shrink=args.max_clip_shrink,
        max_clip_tokens=args.max_clip_tokens,
        max_parallel_calibration_samples=args.max_parallel_calibration_samples,
        modules_to_not_convert=tuple(args.modules_to_not_convert),
    )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install light-vllm[validation] before running AWQ PTQ") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    texts = iter_calibration_texts(
        args.calibration_data,
        text_field=args.calibration_text_field,
    )
    input_ids = build_calibration_input_ids(
        tokenizer,
        texts,
        max_samples=args.max_calibration_samples,
        sequence_length=args.calibration_sequence_length,
    )
    dtype = _DTYPES[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    ).eval()
    if getattr(model.config, "model_type", None) != "qwen2":
        raise ValueError("the first AWQ PTQ adapter supports only Qwen2/Qwen2.5")
    model.config.use_cache = False
    hidden_states, layer_kwargs = capture_qwen2_first_layer_inputs(
        model,
        input_ids,
        device=device,
        max_parallel_samples=config.max_parallel_calibration_samples,
    )
    quantized_modules, reports = quantize_qwen2_layers(
        model,
        hidden_states,
        layer_kwargs,
        config,
        device=device,
    )
    model._light_vllm_modules_to_not_convert = tuple(args.modules_to_not_convert)
    calibration = {
        "source": str(args.calibration_data.resolve()),
        "text_field": args.calibration_text_field,
        "samples": input_ids.shape[0],
        "sequence_length": input_ids.shape[1],
        "tokens": input_ids.numel(),
        "token_ids_sha256": hashlib.sha256(input_ids.numpy().tobytes()).hexdigest(),
        "seed": args.seed,
    }
    with atomic_output_directory(args.output) as staging:
        export_awq_checkpoint(
            model,
            staging,
            quantized_modules,
            group_size=args.group_size,
            reports=reports,
            ptq_config=config,
            calibration=calibration,
            max_shard_bytes=int(args.max_shard_size_gib * 1024**3),
        )
        copy_qwen2_checkpoint_assets(args.model, staging)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "calibration_samples": input_ids.shape[0],
                "calibration_tokens": input_ids.numel(),
                "quantized_modules": len(quantized_modules),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
