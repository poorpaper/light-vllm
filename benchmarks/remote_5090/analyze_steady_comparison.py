from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_preemption_comparison import _load, _median_runs, _mismatch_count, _run

MODES = ("light-strict", "light-rolling10", "vllm-preempt16")
LABELS = {
    "light-strict": "Light strict",
    "light-rolling10": "Light rolling 10%",
    "vllm-preempt16": "vLLM default",
}


def _percent_change(before: float, after: float) -> float:
    return (after / before - 1.0) * 100.0


def _plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    median = summary["median"]
    labels = [LABELS[mode] for mode in MODES]
    x = np.arange(len(MODES))
    width = 0.34

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10.2))
    figure.suptitle("Steady Poisson traffic at 2 requests/s", fontsize=20)

    ttft_p50 = [median[mode]["ttft_p50_s"] for mode in MODES]
    ttft_p95 = [median[mode]["ttft_p95_s"] for mode in MODES]
    axes[0, 0].bar(x - width / 2, ttft_p50, width, label="P50", color="#619CFF")
    axes[0, 0].bar(x + width / 2, ttft_p95, width, label="P95", color="#D89045")
    axes[0, 0].set_ylabel("Seconds")
    axes[0, 0].set_title("Time to first token")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].legend()

    tpot_p95 = [median[mode]["tpot_p95_s"] for mode in MODES]
    bars = axes[0, 1].bar(x, tpot_p95, color=("#355F8A", "#4AA3A2", "#B4474E"))
    axes[0, 1].bar_label(bars, fmt="%.4f", padding=3)
    axes[0, 1].set_ylabel("Seconds per output token")
    axes[0, 1].set_title("TPOT P95")
    axes[0, 1].set_xticks(x, labels)

    gap_p95 = [median[mode]["max_itl_p95_s"] for mode in MODES]
    gap_max = [median[mode]["max_itl_max_s"] for mode in MODES]
    axes[1, 0].bar(x - width / 2, gap_p95, width, label="P95 request", color="#7C62A3")
    axes[1, 0].bar(x + width / 2, gap_max, width, label="Worst request", color="#CE6A85")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Seconds (log scale)")
    axes[1, 0].set_title("Largest in-stream token gap")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].legend()

    e2e_p50 = [median[mode]["e2e_p50_s"] for mode in MODES]
    e2e_p95 = [median[mode]["e2e_p95_s"] for mode in MODES]
    axes[1, 1].bar(x - width / 2, e2e_p50, width, label="P50", color="#619CFF")
    axes[1, 1].bar(x + width / 2, e2e_p95, width, label="P95", color="#D89045")
    axes[1, 1].set_ylabel("Seconds")
    axes[1, 1].set_title("End-to-end request latency")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].legend()

    figure.text(
        0.5,
        0.015,
        "64 identical production-length requests, 5804 prompt + 7514 output tokens; "
        "4096 KV tokens; median of 3 post-warmup runs. All runs: 0 preemptions, "
        "0 KV-capacity pauses, 0 rollbacks.",
        ha="center",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    runs = {mode: [_run(args.input_dir, mode, run) for run in range(1, 4)] for mode in MODES}
    for mode, mode_runs in runs.items():
        for run in mode_runs:
            payload = _load(args.input_dir / f"{mode}-r{run['run']}.json")
            metadata = payload["metadata"]
            if (
                run["requests"] != 64
                or run["successful_requests"] != 64
                or run["failed_requests"] != 0
                or run["output_tokens"] != 7514
                or run["replay_target_hits"] != 64
                or metadata["arrival_mode"] != "poisson"
                or float(metadata["request_rate"]) != 2.0
            ):
                raise ValueError(f"unmatched steady workload in {mode} r{run['run']}")

    median = {mode: _median_runs(mode_runs) for mode, mode_runs in runs.items()}
    strict = median["light-strict"]
    rolling = median["light-rolling10"]
    vllm = median["vllm-preempt16"]
    for mode in MODES:
        scheduler = median[mode]["scheduler"]
        if (
            scheduler["nonpreemptive_pauses"] != 0
            or scheduler["self_resubmits"] != 0
            or scheduler["rolled_back_tokens"] != 0
            or scheduler["vllm_preemptions"] != 0
        ):
            raise ValueError(f"unexpected pressure event in steady mode: {mode}")

    reference = _load(args.input_dir / "light-strict-r1.json")
    summary = {
        "scope": {
            "device": "NVIDIA GeForce RTX 5090 32 GB",
            "model": "Qwen2.5-Coder-7B-Instruct",
            "dtype": "bfloat16",
            "arrival": "Poisson 2 requests/s, seed 20260820",
            "offered_output_rate_tokens_per_s": 2.0 * 117.40625,
            "kv_tokens": 4096,
            "max_num_scheduled_tokens": 512,
            "prompt_tokens": 5804,
            "output_tokens": 7514,
            "runs": "one full warmup then three measured runs per mode",
            "throughput_note": (
                "At sustainable open-loop load, observed output tokens/s is arrival-limited "
                "and is not used to rank engine capacity."
            ),
        },
        "runs": runs,
        "median": median,
        "changes_percent": {
            "light_strict_to_rolling10": {
                "ttft_p50": _percent_change(strict["ttft_p50_s"], rolling["ttft_p50_s"]),
                "ttft_p95": _percent_change(strict["ttft_p95_s"], rolling["ttft_p95_s"]),
                "tpot_p95": _percent_change(strict["tpot_p95_s"], rolling["tpot_p95_s"]),
                "max_itl_p95": _percent_change(strict["max_itl_p95_s"], rolling["max_itl_p95_s"]),
                "e2e_p50": _percent_change(strict["e2e_p50_s"], rolling["e2e_p50_s"]),
                "e2e_p95": _percent_change(strict["e2e_p95_s"], rolling["e2e_p95_s"]),
            },
            "light_rolling10_to_vllm_default": {
                "ttft_p50": _percent_change(rolling["ttft_p50_s"], vllm["ttft_p50_s"]),
                "ttft_p95": _percent_change(rolling["ttft_p95_s"], vllm["ttft_p95_s"]),
                "tpot_p95": _percent_change(rolling["tpot_p95_s"], vllm["tpot_p95_s"]),
                "max_itl_p95": _percent_change(rolling["max_itl_p95_s"], vllm["max_itl_p95_s"]),
                "e2e_p50": _percent_change(rolling["e2e_p50_s"], vllm["e2e_p50_s"]),
                "e2e_p95": _percent_change(rolling["e2e_p95_s"], vllm["e2e_p95_s"]),
            },
        },
        "pressure_events": {mode: median[mode]["scheduler"] for mode in MODES},
        "output_reproducibility": {
            "r1_mismatches_against_light_strict": {
                mode: _mismatch_count(reference, _load(args.input_dir / f"{mode}-r1.json"))
                for mode in MODES
            },
            "note": "Batch-shape-dependent BF16 token IDs are not used as the timing gate.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(summary, args.figure)
    print(json.dumps(summary["changes_percent"], indent=2))


if __name__ == "__main__":
    main()
