from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

IMPORTANT_METRICS = (
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "light_vllm_self_resubmits_total",
    "light_vllm_self_resubmit_rolled_back_tokens_total",
    "light_vllm_nonpreemptive_pauses_total",
    "light_vllm_completion_claim_handoffs_total",
    "light_vllm_speculation_attempts_total",
    "light_vllm_speculative_proposed_nodes_total",
    "light_vllm_speculative_accepted_nodes_total",
    "light_vllm_speculative_verified_tokens_total",
    "light_vllm_speculative_compacted_tokens_total",
)


def _prometheus_totals(path: Path) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    if not path.exists():
        return totals
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([^\s{]+)(?:\{[^}]*\})?\s+([^\s]+)$", line)
        if match is None:
            continue
        name, raw_value = match.groups()
        if name.endswith("_created"):
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            totals[name] += value
    return totals


def _metrics_delta(result_path: Path) -> dict[str, float]:
    before = _prometheus_totals(result_path.with_suffix(".metrics-before.prom"))
    after = _prometheus_totals(result_path.with_suffix(".metrics-after.prom"))
    return {name: after.get(name, 0.0) - before.get(name, 0.0) for name in IMPORTANT_METRICS}


def _load_rows(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        result_path = Path(case["file"])
        if not result_path.is_absolute():
            result_path = manifest_path.parent / result_path
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        metrics = _metrics_delta(result_path)
        attempts = metrics["light_vllm_speculation_attempts_total"]
        proposed = metrics["light_vllm_speculative_proposed_nodes_total"]
        vllm_drafts = metrics["vllm:spec_decode_num_drafts_total"]
        vllm_draft_tokens = metrics["vllm:spec_decode_num_draft_tokens_total"]
        vllm_accepted_tokens = metrics["vllm:spec_decode_num_accepted_tokens_total"]
        row = {
            **case,
            "output_tokens_per_s": summary["output_tokens_per_s"],
            "output_tokens": summary["output_tokens"],
            "output_tokens_mean": summary.get("output_tokens_mean"),
            "output_tokens_p50": summary.get("output_tokens_p50"),
            "output_tokens_p95": summary.get("output_tokens_p95"),
            "output_limit_hits": summary.get("output_limit_hits"),
            "replay_target_hits": summary.get("replay_target_hits"),
            "ttft_p50_s": summary["ttft_p50_s"],
            "ttft_p95_s": summary["ttft_p95_s"],
            "ttft_p99_s": summary["ttft_p99_s"],
            "tpot_p50_s": summary["tpot_p50_s"],
            "max_itl_p95_s": summary["max_itl_p95_s"],
            "max_itl_max_s": summary["max_itl_max_s"],
            "e2e_p50_s": summary["e2e_p50_s"],
            "e2e_p99_s": summary["e2e_p99_s"],
            "e2e_mean_s": summary["e2e_mean_s"],
            "short_ttft_p99_s": (summary.get("by_kind", {}).get("short", {})).get("ttft_p99_s"),
            "long_e2e_p99_s": (summary.get("by_kind", {}).get("long", {})).get("e2e_p99_s"),
            "preemptions": metrics["vllm:num_preemptions_total"],
            "self_resubmits": metrics["light_vllm_self_resubmits_total"],
            "rolled_back_tokens": metrics["light_vllm_self_resubmit_rolled_back_tokens_total"],
            "nonpreemptive_pauses": metrics["light_vllm_nonpreemptive_pauses_total"],
            "completion_claim_handoffs": metrics["light_vllm_completion_claim_handoffs_total"],
            "recompute_events": (
                metrics["vllm:num_preemptions_total"] + metrics["light_vllm_self_resubmits_total"]
            ),
            "mean_verified_tokens": (
                metrics["light_vllm_speculative_verified_tokens_total"] / attempts
                if attempts
                else None
            ),
            "mean_proposed_nodes": (proposed / attempts if attempts else None),
            "mean_accepted_draft_tokens": (
                metrics["light_vllm_speculative_accepted_nodes_total"] / attempts
                if attempts
                else vllm_accepted_tokens / vllm_drafts
                if vllm_drafts
                else None
            ),
            "draft_acceptance_ratio": (
                metrics["light_vllm_speculative_accepted_nodes_total"] / proposed
                if proposed
                else vllm_accepted_tokens / vllm_draft_tokens
                if vllm_draft_tokens
                else None
            ),
        }
        rows.append(row)
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float | None]) -> float:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else 0.0


def _spread(values: list[float | None]) -> float:
    present = [value for value in values if value is not None]
    return statistics.stdev(present) if len(present) > 1 else 0.0


def _group_rows(rows: list[dict[str, Any]], group: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["group"] == group:
            grouped[row["label"]].append(row)
    return grouped


def _bar_chart(
    *,
    grouped: dict[str, list[dict[str, Any]]],
    metrics: tuple[str, ...],
    titles: tuple[str, ...],
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels = list(grouped)
    if len(metrics) == 4:
        figure, axes_grid = plt.subplots(2, 2, figsize=(14, 9))
        axes = axes_grid.ravel().tolist()
    else:
        figure, axes_grid = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.8))
        axes = [axes_grid] if len(metrics) == 1 else axes_grid
    colors = ["#2f6bff", "#ef8354", "#3ca370", "#8d6cab", "#d6a21f"]
    for axis, metric, title in zip(axes, metrics, titles, strict=True):
        values = [_mean([row.get(metric) for row in grouped[label]]) for label in labels]
        errors = [_spread([row.get(metric) for row in grouped[label]]) for label in labels]
        bars = axis.bar(
            labels,
            values,
            yerr=errors,
            capsize=4,
            color=colors[: len(labels)],
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=20)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def _plot_speculation(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    grouped = _group_rows(rows, "speculation")
    labels = list(grouped)
    throughput = {
        label: _mean([row["output_tokens_per_s"] for row in grouped[label]]) for label in labels
    }
    throughput_spread = {
        label: _spread([row["output_tokens_per_s"] for row in grouped[label]]) for label in labels
    }
    light_baseline = throughput.get("light-vllm no spec", 1.0)
    vllm_baseline = throughput.get("vLLM no spec", 1.0)
    speedups = [
        throughput[label] / (vllm_baseline if label.startswith("vLLM") else light_baseline)
        for label in labels
    ]
    speedup_errors = [
        throughput_spread[label] / (vllm_baseline if label.startswith("vLLM") else light_baseline)
        for label in labels
    ]
    verified = []
    target_work_per_visible_token = []
    for label in labels:
        mean_verified = _mean([row.get("mean_verified_tokens") for row in grouped[label]])
        mean_proposed = _mean([row.get("mean_proposed_nodes") for row in grouped[label]])
        if mean_verified == 0.0:
            mean_verified = 1.0
        verified.append(mean_verified)
        target_work_per_visible_token.append((1.0 + mean_proposed) / mean_verified)

    figure, axes = plt.subplots(1, 4, figsize=(24, 4.8))
    colors = ["#2f6bff", "#ef8354", "#3ca370", "#8d6cab", "#d6a21f"]
    throughput_bars = axes[0].bar(
        labels,
        [throughput[label] for label in labels],
        yerr=[throughput_spread[label] for label in labels],
        capsize=4,
        color=colors[: len(labels)],
    )
    axes[0].set_title("Speculative workload: output throughput")
    axes[0].set_ylabel("output tokens/s")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].bar_label(throughput_bars, fmt="%.1f", padding=3, fontsize=8)

    speedup_bars = axes[1].bar(
        labels,
        speedups,
        yerr=speedup_errors,
        capsize=4,
        color=colors[: len(labels)],
    )
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1].set_title("Throughput speedup vs own baseline")
    axes[1].set_ylabel("relative speedup")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].bar_label(speedup_bars, fmt="%.2fx", padding=3, fontsize=8)

    verified_bars = axes[2].bar(labels, verified, color=colors[: len(labels)])
    axes[2].set_title("Visible tokens per target verification")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].bar_label(verified_bars, fmt="%.2f", padding=3, fontsize=8)

    efficiency_bars = axes[3].bar(
        labels,
        target_work_per_visible_token,
        color=colors[: len(labels)],
    )
    axes[3].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[3].set_title("Target query slots per visible token (lower is better)")
    axes[3].tick_params(axis="x", rotation=20)
    axes[3].bar_label(efficiency_bars, fmt="%.2f", padding=3, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def _plot_profile_breakdown(profile_path: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    totals = {
        "Light before wall": profile["light_before"]["wall"],
        "Light before CUDA": profile["light_before"]["cuda_total"],
        "Light optimized wall": profile["light_optimized"]["wall"],
        "Light optimized CUDA": profile["light_optimized"]["cuda_total"],
        "vLLM eager CUDA range": profile["vllm_eager"]["execute_model_cuda_range"],
    }
    kernels = {
        "Light GEMM": profile["light_before"]["gemm"],
        "vLLM GEMM": profile["vllm_eager"]["gemm"],
        "Light attention": profile["light_before"]["paged_attention"],
        "vLLM attention": profile["vllm_eager"]["flash_attention"],
    }
    colors = ["#2f6bff", "#7398ff", "#ef8354", "#f4aa83", "#3ca370"]
    figure, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    total_bars = axes[0].bar(list(totals), list(totals.values()), color=colors)
    axes[0].set_title("Decode step latency (measurement boundaries differ)")
    axes[0].set_ylabel("milliseconds per step")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].bar_label(total_bars, fmt="%.2f", padding=3, fontsize=8)
    kernel_bars = axes[1].bar(
        list(kernels),
        list(kernels.values()),
        color=("#2f6bff", "#3ca370", "#7398ff", "#76c59b"),
    )
    axes[1].set_title("Matched CUDA kernel categories")
    axes[1].set_ylabel("milliseconds per step")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].bar_label(kernel_bars, fmt="%.3f", padding=3, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args.manifest)
    _write_csv(rows, args.output_dir / "summary.csv")
    profile_path = args.manifest.parent / "profiling_breakdown.json"
    if profile_path.exists():
        _plot_profile_breakdown(profile_path, args.output_dir / "00_decode_time_breakdown.png")
    baseline = _group_rows(rows, "baseline")
    if baseline:
        _bar_chart(
            grouped=baseline,
            metrics=("output_tokens_per_s", "e2e_p99_s"),
            titles=("Same-GPU output throughput", "Same-GPU p99 end-to-end latency"),
            output=args.output_dir / "01_system_baseline.png",
        )
    scheduling = _group_rows(rows, "scheduling")
    if scheduling:
        _bar_chart(
            grouped=scheduling,
            metrics=(
                "output_tokens_per_s",
                "ttft_p50_s",
                "max_itl_max_s",
                "recompute_events",
            ),
            titles=(
                "KV pressure: output throughput",
                "KV pressure: median TTFT",
                "KV pressure: worst token-stream gap",
                "KV pressure: preemption/self-resubmit events",
            ),
            output=args.output_dir / "02_scheduling_pressure.png",
        )
    production = _group_rows(rows, "production")
    if production:
        _bar_chart(
            grouped=production,
            metrics=(
                "output_tokens_per_s",
                "ttft_p50_s",
                "ttft_p95_s",
                "max_itl_max_s",
            ),
            titles=(
                "Production trace: output throughput",
                "Production trace: median TTFT",
                "Production trace: p95 TTFT",
                "Production trace: worst token-stream gap",
            ),
            output=args.output_dir / "04_production_trace.png",
        )
    production_burst = _group_rows(rows, "production_burst")
    if production_burst:
        _bar_chart(
            grouped=production_burst,
            metrics=(
                "output_tokens_per_s",
                "ttft_p50_s",
                "max_itl_max_s",
                "recompute_events",
            ),
            titles=(
                "Production burst: output throughput",
                "Production burst: median TTFT",
                "Production burst: worst token-stream gap",
                "Production burst: preemption / self-resubmit events",
            ),
            output=args.output_dir / "05_production_burst.png",
        )
    if _group_rows(rows, "speculation"):
        _plot_speculation(rows, args.output_dir / "03_speculation.png")


if __name__ == "__main__":
    main()
