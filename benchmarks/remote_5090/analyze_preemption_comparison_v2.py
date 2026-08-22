from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_preemption_comparison import (
    _load,
    _median_runs,
    _mismatch_count,
    _percent_change,
    _run,
)

MODES = (
    "light-strict",
    "light-optimistic",
    "vllm-nopreempt6",
    "vllm-preempt16",
)
LABELS = {
    "light-strict": "Light strict",
    "light-optimistic": "Light optimistic",
    "vllm-nopreempt6": "vLLM no-preempt (6)",
    "vllm-preempt16": "vLLM default (16)",
}
COLORS = ("#355F8A", "#4AA3A2", "#C57B36", "#B4474E")


def _scenario(input_dir: Path, *, request_rate: float) -> dict[str, Any]:
    runs = {mode: [_run(input_dir, mode, run) for run in range(1, 4)] for mode in MODES}
    for mode, mode_runs in runs.items():
        for run in mode_runs:
            payload = _load(input_dir / f"{mode}-r{run['run']}.json")
            metadata = payload["metadata"]
            if (
                run["requests"] != 64
                or run["successful_requests"] != 64
                or run["failed_requests"] != 0
                or run["output_tokens"] != 7514
                or run["replay_target_hits"] != 64
                or metadata["arrival_mode"] != "poisson"
                or float(metadata["request_rate"]) != request_rate
            ):
                raise ValueError(f"unmatched workload in {input_dir.name}/{mode} r{run['run']}")

    median = {mode: _median_runs(mode_runs) for mode, mode_runs in runs.items()}
    reference = _load(input_dir / "light-strict-r1.json")
    return {
        "request_rate": request_rate,
        "runs": runs,
        "median": median,
        "output_mismatches_against_light_strict_r1": {
            mode: _mismatch_count(reference, _load(input_dir / f"{mode}-r1.json")) for mode in MODES
        },
        "changes_percent": {
            "strict_to_optimistic": {
                "throughput": _percent_change(
                    median["light-strict"]["throughput_tokens_per_s"],
                    median["light-optimistic"]["throughput_tokens_per_s"],
                ),
                "ttft_p95": _percent_change(
                    median["light-strict"]["ttft_p95_s"],
                    median["light-optimistic"]["ttft_p95_s"],
                ),
                "tpot_p95": _percent_change(
                    median["light-strict"]["tpot_p95_s"],
                    median["light-optimistic"]["tpot_p95_s"],
                ),
                "max_itl_p95": _percent_change(
                    median["light-strict"]["max_itl_p95_s"],
                    median["light-optimistic"]["max_itl_p95_s"],
                ),
                "e2e_p95": _percent_change(
                    median["light-strict"]["e2e_p95_s"],
                    median["light-optimistic"]["e2e_p95_s"],
                ),
            },
            "vllm_nopreempt_to_default": {
                "throughput": _percent_change(
                    median["vllm-nopreempt6"]["throughput_tokens_per_s"],
                    median["vllm-preempt16"]["throughput_tokens_per_s"],
                ),
                "ttft_p95": _percent_change(
                    median["vllm-nopreempt6"]["ttft_p95_s"],
                    median["vllm-preempt16"]["ttft_p95_s"],
                ),
                "max_itl_p95": _percent_change(
                    median["vllm-nopreempt6"]["max_itl_p95_s"],
                    median["vllm-preempt16"]["max_itl_p95_s"],
                ),
                "e2e_p95": _percent_change(
                    median["vllm-nopreempt6"]["e2e_p95_s"],
                    median["vllm-preempt16"]["e2e_p95_s"],
                ),
            },
        },
    }


def _plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 5, figsize=(23, 9.5))
    figure.suptitle("Production-length Poisson traffic: scheduling trade-offs", fontsize=20)
    x = np.arange(len(MODES))

    for row, (scenario_name, scenario) in enumerate(summary["scenarios"].items()):
        median = scenario["median"]
        title = f"{scenario_name}: {scenario['request_rate']:.0f} req/s"
        metrics = (
            ("throughput_tokens_per_s", "Output tok/s", False),
            ("ttft_p95_s", "TTFT P95 (s)", False),
            ("tpot_p95_s", "TPOT P95 (s)", False),
            ("max_itl_p95_s", "Max token gap P95 (s)", True),
        )
        for column, (key, label, log_scale) in enumerate(metrics):
            values = [median[mode][key] for mode in MODES]
            bars = axes[row, column].bar(x, values, color=COLORS)
            axes[row, column].bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
            axes[row, column].set_ylabel(label)
            axes[row, column].set_title(f"{title}\n{label}")
            axes[row, column].set_xticks(x, [LABELS[mode] for mode in MODES], rotation=18)
            if log_scale:
                axes[row, column].set_yscale("log")

        resubmits = [median[mode]["scheduler"]["self_resubmits"] for mode in MODES]
        preemptions = [median[mode]["scheduler"]["vllm_preemptions"] for mode in MODES]
        width = 0.36
        axes[row, 4].bar(
            x - width / 2,
            resubmits,
            width,
            label="Light self-resubmits",
            color="#4AA3A2",
        )
        axes[row, 4].bar(
            x + width / 2,
            preemptions,
            width,
            label="vLLM preemptions",
            color="#B4474E",
        )
        axes[row, 4].set_ylabel("Events/run")
        axes[row, 4].set_title(f"{title}\nCapacity-pressure events")
        axes[row, 4].set_xticks(x, [LABELS[mode] for mode in MODES], rotation=18)
        axes[row, 4].legend(fontsize=8)

    figure.text(
        0.5,
        0.01,
        "RTX 5090 · Qwen2.5-Coder-7B BF16 · 4096 KV tokens · batch budget 512 · "
        "same 64 prompts and 7514 replayed output tokens · median of 3 post-warmup runs",
        ha="center",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steady-dir", type=Path, required=True)
    parser.add_argument("--pressure-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    summary = {
        "scope": {
            "device": "NVIDIA GeForce RTX 5090 32 GB",
            "model": "Qwen2.5-Coder-7B-Instruct",
            "dtype": "bfloat16",
            "block_size": 16,
            "kv_tokens": 4096,
            "max_num_scheduled_tokens": 512,
            "prefix_cache": "enabled for light; required by optimistic self-resubmit",
            "speculative_decoding": False,
            "ttft_admission": False,
            "arrival": "Poisson, seed 20260820",
            "prompt_tokens": 5804,
            "output_tokens": 7514,
            "runs": "one full warmup then three measured runs per mode and rate",
        },
        "scenarios": {
            "steady": _scenario(args.steady_dir, request_rate=2.0),
            "pressure": _scenario(args.pressure_dir, request_rate=8.0),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(summary, args.figure)
    print(
        json.dumps(
            {name: data["changes_percent"] for name, data in summary["scenarios"].items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
