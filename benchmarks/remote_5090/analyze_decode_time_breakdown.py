from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _kernel_category(name: str) -> str:
    lowered = name.lower()
    if "cutlass::kernel2" in lowered or "gemvx" in lowered:
        return "GEMM"
    if "paged_attention" in lowered or "flash_fwd" in lowered:
        return "Attention"
    if "rms_norm" in lowered or "layer_norm" in lowered:
        return "RMSNorm/residual"
    if "silu" in lowered or "act_and_mul" in lowered:
        return "SiLU/mul"
    if "rotary" in lowered:
        return "RoPE"
    if "reshape_and_cache" in lowered or "index_copy_kernel_impl" in lowered:
        return "KV cache write"
    if any(
        fragment in lowered
        for fragment in ("gumbel", "argmax", "sampled", "rejected", "combine_sample")
    ):
        return "Sampling"
    if "direct_copy" in lowered:
        return "Tensor copies"
    if any(
        fragment in lowered
        for fragment in (
            "gather_block",
            "slot_mapping",
            "prepare_pos",
            "apply_write",
            "post_update",
        )
    ):
        return "Runner metadata"
    if "indexselect" in lowered and "bfloat16" in lowered:
        return "Embedding/RoPE lookup"
    return "Other kernels"


def _profile_steps(events: list[dict[str, Any]], default: int) -> int:
    annotations = [
        event
        for event in events
        if event.get("cat") == "gpu_user_annotation"
        and str(event.get("name", "")).startswith("execute_")
    ]
    return len(annotations) or default


def _trace_summary(path: Path, *, default_steps: int) -> dict[str, Any]:
    events = json.loads(path.read_text(encoding="utf-8"))["traceEvents"]
    steps = _profile_steps(events, default_steps)
    categories: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    launch_calls = 0
    launch_cpu_us = 0.0
    memcpy_count = 0
    memcpy_gpu_us = 0.0
    for event in events:
        name = str(event.get("name", ""))
        duration = float(event.get("dur", 0.0))
        if event.get("cat") == "kernel":
            category = _kernel_category(name)
            categories[category][0] += 1
            categories[category][1] += duration
        if name in {"cudaLaunchKernel", "cuLaunchKernel", "cuLaunchKernelEx"}:
            launch_calls += 1
            launch_cpu_us += duration
        if event.get("cat") == "gpu_memcpy":
            memcpy_count += 1
            memcpy_gpu_us += duration

    per_category = {
        name: {
            "launches_per_step": values[0] / steps,
            "gpu_ms_per_step": values[1] / 1000 / steps,
        }
        for name, values in sorted(
            categories.items(),
            key=lambda item: item[1][1],
            reverse=True,
        )
    }
    return {
        "trace": str(path),
        "profile_steps": steps,
        "kernel_launches_per_step": sum(
            values["launches_per_step"] for values in per_category.values()
        ),
        "kernel_gpu_ms_per_step": sum(
            values["gpu_ms_per_step"] for values in per_category.values()
        ),
        "launch_api_calls_per_step": launch_calls / steps,
        "launch_api_cpu_ms_per_step_profiled": launch_cpu_us / 1000 / steps,
        "gpu_memcpy_count_per_step": memcpy_count / steps,
        "gpu_memcpy_ms_per_step": memcpy_gpu_us / 1000 / steps,
        "categories": per_category,
    }


def _prometheus_value(path: Path, metric: str, *, model_tokens: int) -> float:
    pattern = re.compile(
        rf'^{re.escape(metric)}\{{[^}}]*model_tokens_computed_le="{model_tokens}"[^}}]*\}} (.+)$'
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            return float(match.group(1))
    raise ValueError(f"missing {metric} for model_tokens={model_tokens} in {path}")


def _prometheus_bucket_values(path: Path, metric: str) -> dict[str, float]:
    pattern = re.compile(
        rf'^{re.escape(metric)}\{{[^}}]*model_tokens_computed_le="([^"]+)"[^}}]*\}} (.+)$'
    )
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            values[match.group(1)] = float(match.group(2))
    if not values:
        raise ValueError(f"missing {metric} buckets in {path}")
    return values


def _engine_device_step_ms(path: Path, *, model_tokens: int) -> float:
    stem = path.with_suffix("")
    before = Path(f"{stem}.metrics-before.prom")
    after = Path(f"{stem}.metrics-after.prom")
    count_metric = "light_vllm_engine_step_seconds_count"
    sum_metric = "light_vllm_engine_step_seconds_sum"
    count = _prometheus_value(after, count_metric, model_tokens=model_tokens) - _prometheus_value(
        before, count_metric, model_tokens=model_tokens
    )
    duration = _prometheus_value(after, sum_metric, model_tokens=model_tokens) - _prometheus_value(
        before, sum_metric, model_tokens=model_tokens
    )
    if count <= 0:
        raise ValueError(f"non-positive engine step count delta for {path}: {count}")
    return duration / count * 1000


def _amortized_engine_device_step_ms(path: Path, *, output_steps: float) -> float:
    stem = path.with_suffix("")
    before = Path(f"{stem}.metrics-before.prom")
    after = Path(f"{stem}.metrics-after.prom")
    metric = "light_vllm_engine_step_seconds_sum"
    before_values = _prometheus_bucket_values(before, metric)
    after_values = _prometheus_bucket_values(after, metric)
    duration = sum(value - before_values.get(bucket, 0.0) for bucket, value in after_values.items())
    if output_steps <= 0:
        raise ValueError(f"non-positive output step count for {path}: {output_steps}")
    return duration / output_steps * 1000


def _service_summary(
    paths: list[Path],
    *,
    batch_size: int,
    engine_metric_tokens: int | None = None,
) -> dict[str, Any]:
    runs = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        prompt_tokens = [int(result["prompt_tokens"]) for result in payload["results"]]
        output_tokens = [int(result["output_tokens"]) for result in payload["results"]]
        throughput = float(summary["output_tokens_per_s"])
        run = {
            "file": str(path),
            "throughput_tokens_per_s": throughput,
            "effective_batch_step_ms": batch_size / throughput * 1000,
            "ttft_p50_ms": float(summary["ttft_p50_s"]) * 1000,
            "tpot_p50_ms": float(summary["tpot_p50_s"]) * 1000,
            "prompt_tokens": statistics.fmean(prompt_tokens),
            "output_tokens": statistics.fmean(output_tokens),
        }
        if engine_metric_tokens is not None:
            run["engine_decode_device_step_ms"] = _engine_device_step_ms(
                path, model_tokens=engine_metric_tokens
            )
            output_steps = sum(output_tokens) / batch_size
            run["engine_amortized_device_step_ms"] = _amortized_engine_device_step_ms(
                path, output_steps=output_steps
            )
        runs.append(run)
    fields = (
        "throughput_tokens_per_s",
        "effective_batch_step_ms",
        "ttft_p50_ms",
        "tpot_p50_ms",
        "prompt_tokens",
        "output_tokens",
    )
    if engine_metric_tokens is not None:
        fields += (
            "engine_decode_device_step_ms",
            "engine_amortized_device_step_ms",
        )
    return {
        "runs": runs,
        "mean": {field: statistics.fmean(run[field] for run in runs) for field in fields},
    }


def _plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    systems = ("light-vllm", "vLLM eager")
    category_order = (
        "GEMM",
        "Attention",
        "RMSNorm/residual",
        "KV cache write",
        "SiLU/mul",
        "RoPE",
        "Tensor copies",
        "Sampling",
        "Runner metadata",
        "Embedding/RoPE lookup",
        "Other kernels",
    )
    colors = plt.get_cmap("tab20").colors
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)

    bottoms = [0.0, 0.0]
    for index, category in enumerate(category_order):
        values = [
            summary[system]["trace"]["categories"].get(category, {}).get("gpu_ms_per_step", 0.0)
            for system in systems
        ]
        if not any(values):
            continue
        axes[0].bar(systems, values, bottom=bottoms, label=category, color=colors[index])
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    axes[0].set_title("Fixed decode GPU kernels\nbatch=16, context=256, query=1")
    axes[0].set_ylabel("GPU time per step (ms)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(
        fontsize=7,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
    )
    for position, value in enumerate(bottoms):
        axes[0].text(position, value, f"{value:.3f}", ha="center", va="bottom")

    kernel_times = [summary[system]["trace"]["kernel_gpu_ms_per_step"] for system in systems]
    service_times = [
        summary[system]["service"]["mean"]["effective_batch_step_ms"] for system in systems
    ]
    light_device = summary["light-vllm"]["service"]["mean"]["engine_amortized_device_step_ms"]
    device_residuals = [max(light_device - kernel_times[0], 0.0), 0.0]
    outside_executor = [
        max(service_times[0] - light_device, 0.0),
        max(service_times[1] - kernel_times[1], 0.0),
    ]
    axes[1].bar(systems, kernel_times, label="Profiled GPU kernel sum", color="#4C78A8")
    axes[1].bar(
        systems,
        device_residuals,
        bottom=kernel_times,
        label="light device-boundary residual*",
        color="#F58518",
    )
    executor_bottoms = [
        kernel + residual for kernel, residual in zip(kernel_times, device_residuals, strict=True)
    ]
    axes[1].bar(
        systems,
        outside_executor,
        bottom=executor_bottoms,
        label="Outside executor / unattributed*",
        color="#54A24B",
    )
    axes[1].set_title("Effective service time per 16-token decode step")
    axes[1].set_ylabel("Milliseconds")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    for position, value in enumerate(service_times):
        axes[1].text(position, value, f"{value:.3f}", ha="center", va="bottom")
    axes[1].text(
        0.5,
        -0.16,
        "*Profile and service are separate runs. vLLM has no matching device-boundary metric.",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=8,
    )
    figure.suptitle("Steady decode time breakdown on RTX 5090", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--light-profile", type=Path, required=True)
    parser.add_argument("--light-trace", type=Path, required=True)
    parser.add_argument("--vllm-trace", type=Path, required=True)
    parser.add_argument("--light-service", type=Path, nargs="+", required=True)
    parser.add_argument("--vllm-service", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    light_profile = json.loads(args.light_profile.read_text(encoding="utf-8"))
    summary = {
        "scope": {
            "model": "Qwen2.5-Coder-7B-Instruct",
            "dtype": "bfloat16",
            "device": "RTX 5090",
            "batch_size": args.batch_size,
            "profile_context_tokens": 256,
            "query_tokens": 1,
            "service_prompt_tokens": 16,
            "service_output_tokens": 512,
            "vllm_mode": "eager",
        },
        "notes": [
            "GPU category values are sums of kernel durations, not enclosing CUDA ranges.",
            (
                "Both traces average fixed decode steps after warm-up and include greedy "
                "sampling; see each trace's profile_steps field for the exact count."
            ),
            (
                "Service effective step time is batch_size / output throughput and includes "
                "amortized prefill."
            ),
            (
                "Light's engine device time comes from CUDA events around Executor.execute; "
                "the amortized value includes every engine step in the same service run."
            ),
            (
                "Profiler and service measurements are separate runs, so their difference is a "
                "time-budget residual rather than an exact causal attribution."
            ),
            (
                "The service context grows from a 16-token prompt through 512 outputs; "
                "context=256 is close to its midpoint."
            ),
        ],
        "light-vllm": {
            "trace": _trace_summary(
                args.light_trace,
                default_steps=int(light_profile.get("profile_steps", 1)),
            ),
            "direct_fixed_step_wall_ms": float(light_profile["mean_step_ms"]),
            "service": _service_summary(
                args.light_service,
                batch_size=args.batch_size,
                engine_metric_tokens=args.batch_size,
            ),
        },
        "vLLM eager": {
            "trace": _trace_summary(args.vllm_trace, default_steps=30),
            "direct_model_wall_ms": None,
            "service": _service_summary(args.vllm_service, batch_size=args.batch_size),
        },
    }
    light = summary["light-vllm"]
    vllm = summary["vLLM eager"]
    light_kernel_ms = light["trace"]["kernel_gpu_ms_per_step"]
    vllm_kernel_ms = vllm["trace"]["kernel_gpu_ms_per_step"]
    light_device_ms = light["service"]["mean"]["engine_amortized_device_step_ms"]
    light_service_ms = light["service"]["mean"]["effective_batch_step_ms"]
    vllm_service_ms = vllm["service"]["mean"]["effective_batch_step_ms"]
    summary["derived"] = {
        "service_step_gap_light_minus_vllm_ms": light_service_ms - vllm_service_ms,
        "profiled_kernel_gap_light_minus_vllm_ms": light_kernel_ms - vllm_kernel_ms,
        "light_profiled_kernel_to_direct_fixed_wall_residual_ms": (
            light["direct_fixed_step_wall_ms"] - light_kernel_ms
        ),
        "light_profiled_kernel_to_service_device_residual_ms": light_device_ms - light_kernel_ms,
        "light_outside_executor_residual_ms": light_service_ms - light_device_ms,
        "vllm_profiled_kernel_to_service_residual_ms": vllm_service_ms - vllm_kernel_ms,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "steady_decode_time_breakdown.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(summary, args.output_dir / "figures" / "08_steady_decode_time_breakdown.png")


if __name__ == "__main__":
    main()
