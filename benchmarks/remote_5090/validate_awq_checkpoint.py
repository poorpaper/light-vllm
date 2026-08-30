"""加载一个真实 AWQ checkpoint，并记录首轮 native forward 的正确性事实。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

from light_vllm import ForwardBatch, ModelSpec, create_runner
from light_vllm.modeling.quantization.calibration import (
    build_calibration_input_ids,
    iter_calibration_texts,
)
from light_vllm.runtime.execution.dense_attention import (
    DenseAttentionMetadata,
    TorchDenseAttention,
)
from light_vllm.runtime.execution.layout import linear_query_layout


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dense-model", type=Path)
    parser.add_argument("--prompt", default="Write a short Python function that adds two integers.")
    parser.add_argument("--architecture", default="qwen2.5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("auto", "torch", "cuda"), default="cuda")
    parser.add_argument("--evaluation-data", type=Path)
    parser.add_argument("--evaluation-text-field", default="text")
    parser.add_argument("--evaluation-max-samples", type=int, default=8)
    parser.add_argument("--evaluation-sequence-length", type=int, default=256)
    parser.add_argument("--output", type=Path)
    return parser


def _timed_cuda_call(callable_, *, repeats: int = 10) -> tuple[torch.Tensor, float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = callable_()
    end.record()
    end.synchronize()
    first_ms = start.elapsed_time(end)

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = callable_()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return result, first_ms, statistics.median(samples)


def _nll(logits: torch.Tensor, token_ids: torch.Tensor) -> tuple[float, int]:
    """累计 next-token loss；按总 token 归并，避免对短样本重复取平均。"""

    if logits.shape[:2] != token_ids.shape or token_ids.shape[1] < 2:
        raise ValueError("quality logits and token IDs must contain matching sequences")
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        token_ids[:, 1:].reshape(-1),
        reduction="sum",
    )
    return float(loss.item()), token_ids.shape[0] * (token_ids.shape[1] - 1)


def _quality_result(total_nll: float, tokens: int) -> dict[str, float | int]:
    mean_nll = total_nll / tokens
    return {
        "tokens": tokens,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
    }


def main() -> None:
    args = _parser().parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("this acceptance script records CUDA memory and requires a CUDA device")

    from transformers import AutoTokenizer

    tokenizer_path = args.dense_model or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if not token_ids:
        raise ValueError("prompt produced no token IDs")
    evaluation_ids: torch.Tensor | None = None
    evaluation_metadata: dict[str, object] | None = None
    if args.evaluation_data is not None:
        evaluation_ids = build_calibration_input_ids(
            tokenizer,
            iter_calibration_texts(
                args.evaluation_data,
                text_field=args.evaluation_text_field,
            ),
            max_samples=args.evaluation_max_samples,
            sequence_length=args.evaluation_sequence_length,
        )
        if evaluation_ids.shape[1] < 2:
            raise ValueError("quality evaluation requires at least two tokens per sequence")
        evaluation_metadata = {
            "source": str(args.evaluation_data.resolve()),
            "samples": evaluation_ids.shape[0],
            "sequence_length": evaluation_ids.shape[1],
            "input_tokens": evaluation_ids.numel(),
            "token_ids_sha256": hashlib.sha256(evaluation_ids.numpy().tobytes()).hexdigest(),
        }

    torch.cuda.set_device(device)
    dense_logits: torch.Tensor | None = None
    dense_result: dict[str, object] | None = None
    if args.dense_model is not None:
        from transformers import AutoModelForCausalLM

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        load_started = time.perf_counter()
        dense_model = (
            AutoModelForCausalLM.from_pretrained(
                args.dense_model,
                dtype=torch.float16,
                local_files_only=True,
            )
            .eval()
            .to(device)
        )
        torch.cuda.synchronize(device)
        dense_load_seconds = time.perf_counter() - load_started
        dense_model_memory = torch.cuda.memory_allocated(device)
        dense_tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            dense_output, dense_first_ms, dense_median_ms = _timed_cuda_call(
                lambda: dense_model(input_ids=dense_tokens, use_cache=False).logits[:, -1, :]
            )
        dense_logits = dense_output.detach().float().cpu()
        dense_token = int(torch.argmax(dense_logits[0]).item())
        dense_result = {
            "model": str(args.dense_model.resolve()),
            "next_token_id": dense_token,
            "next_token_text": tokenizer.decode([dense_token]),
            "load_seconds": dense_load_seconds,
            "first_forward_ms": dense_first_ms,
            "forward_ms_median": dense_median_ms,
            "model_memory_mib": dense_model_memory / 1024**2,
            "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        }
        if evaluation_ids is not None:
            dense_nll = 0.0
            dense_quality_tokens = 0
            with torch.inference_mode():
                # 单样本执行保持显存有界，也与 native packed 路径使用相同分块。
                for sample in evaluation_ids:
                    sample = sample.unsqueeze(0).to(device)
                    sample_logits = dense_model(input_ids=sample, use_cache=False).logits
                    loss, count = _nll(sample_logits, sample)
                    dense_nll += loss
                    dense_quality_tokens += count
            dense_result["quality"] = _quality_result(dense_nll, dense_quality_tokens)
            # 不让最后一个评测 logits 污染后续 AWQ 常驻显存读数。
            del sample, sample_logits
        del dense_model, dense_tokens, dense_output
        gc.collect()
        torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(device)
    runner = create_runner()
    load_started = time.perf_counter()
    runner.load(
        ModelSpec(
            architecture=args.architecture,
            loader="safetensors",
            weights=args.model,
            device=device,
            dtype=torch.float16,
            quantization="awq",
            quantization_backend=args.backend,
        )
    )
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    model_memory = torch.cuda.memory_allocated(device)

    session = runner.open_session()
    tokens = torch.tensor(token_ids, dtype=torch.long, device=device)
    positions = torch.arange(tokens.numel(), dtype=torch.long, device=device)

    def awq_forward() -> torch.Tensor:
        # AttentionContext 表达一次 model step 的 KV 生命周期，不能跨重复计时复用。
        attention = TorchDenseAttention(
            session.kv_cache_spec,
            DenseAttentionMetadata(
                positions=positions,
                query_layouts=(linear_query_layout(tokens.numel()),),
            ),
        )
        return session.forward(
            ForwardBatch(
                input_ids=tokens,
                positions=positions,
                attention=attention,
                logit_query_indices=(tokens.numel() - 1,),
            )
        ).logits

    with torch.inference_mode():
        logits, first_forward_ms, median_forward_ms = _timed_cuda_call(awq_forward)
    next_token = int(torch.argmax(logits[0]).item())
    result = {
        "gpu": torch.cuda.get_device_name(device),
        "prompt_tokens": len(token_ids),
        "dense": dense_result,
        "awq": {
            "model": str(args.model.resolve()),
            "backend": args.backend,
            "tensor_parallel_size": session.tensor_parallel_size,
            "next_token_id": next_token,
            "next_token_text": tokenizer.decode([next_token]),
            "load_seconds": load_seconds,
            "first_forward_ms": first_forward_ms,
            "forward_ms_median": median_forward_ms,
            "model_memory_mib": model_memory / 1024**2,
            "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        },
    }
    if evaluation_metadata is not None:
        result["evaluation"] = evaluation_metadata
    if evaluation_ids is not None:
        awq_nll = 0.0
        awq_quality_tokens = 0
        with torch.inference_mode():
            for sample in evaluation_ids:
                sample = sample.to(device)
                sample_positions = torch.arange(sample.numel(), dtype=torch.long, device=device)
                sample_attention = TorchDenseAttention(
                    session.kv_cache_spec,
                    DenseAttentionMetadata(
                        positions=sample_positions,
                        query_layouts=(linear_query_layout(sample.numel()),),
                    ),
                )
                sample_logits = session.forward(
                    ForwardBatch(
                        input_ids=sample,
                        positions=sample_positions,
                        attention=sample_attention,
                    )
                ).logits.unsqueeze(0)
                loss, count = _nll(sample_logits, sample.unsqueeze(0))
                awq_nll += loss
                awq_quality_tokens += count
        awq_quality = _quality_result(awq_nll, awq_quality_tokens)
        result["awq"]["quality"] = awq_quality
        if dense_result is not None and "quality" in dense_result:
            dense_quality = dense_result["quality"]
            dense_ppl = float(dense_quality["perplexity"])
            awq_ppl = float(awq_quality["perplexity"])
            result["quality_comparison"] = {
                "perplexity_delta": awq_ppl - dense_ppl,
                "perplexity_delta_percent": (awq_ppl / dense_ppl - 1) * 100,
            }
    if dense_logits is not None:
        awq_logits = logits.detach().float().cpu()
        difference = awq_logits - dense_logits
        result["comparison"] = {
            "top1_equal": next_token == int(torch.argmax(dense_logits[0]).item()),
            "logits_mean_abs_error": difference.abs().mean().item(),
            "logits_max_abs_error": difference.abs().max().item(),
            "logits_cosine_similarity": torch.nn.functional.cosine_similarity(
                awq_logits, dense_logits
            ).item(),
            "model_memory_ratio": model_memory / (dense_result["model_memory_mib"] * 1024**2),
        }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
