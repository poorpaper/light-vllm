from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

MODES = (
    "light-strict",
    "light-rolling10",
    "vllm-nopreempt6",
    "vllm-preempt16",
)
LABELS = {
    "light-strict": "Light strict",
    "light-rolling10": "Light rolling 10%",
    "vllm-nopreempt6": "vLLM no-preempt (6)",
    "vllm-preempt16": "vLLM preempt (16)",
}
COUNTERS = {
    "nonpreemptive_pauses": "light_vllm_nonpreemptive_pauses_total",
    "completion_claim_handoffs": "light_vllm_completion_claim_handoffs_total",
    "self_resubmits": "light_vllm_self_resubmits_total",
    "rolled_back_tokens": "light_vllm_self_resubmit_rolled_back_tokens_total",
    "vllm_preemptions": "vllm:num_preemptions_total",
    "vllm_iteration_tokens": "vllm:iteration_tokens_total_sum",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prometheus_value(path: Path, metric: str) -> float | None:
    pattern = re.compile(
        rf"^{re.escape(metric)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$",
        re.MULTILINE,
    )
    match = pattern.search(path.read_text(encoding="utf-8"))
    return float(match.group(1)) if match is not None else None


def _counter_delta(input_dir: Path, prefix: str, metric: str) -> float:
    before = _prometheus_value(input_dir / f"{prefix}.metrics-before.prom", metric)
    after = _prometheus_value(input_dir / f"{prefix}.metrics-after.prom", metric)
    if before is None or after is None:
        return 0.0
    return after - before


def _run(input_dir: Path, mode: str, run: int) -> dict[str, Any]:
    prefix = f"{mode}-r{run}"
    payload = _load(input_dir / f"{prefix}.json")
    summary = payload["summary"]
    return {
        "run": run,
        "requests": int(summary["requests"]),
        "successful_requests": int(summary["successful_requests"]),
        "failed_requests": int(summary["failed_requests"]),
        "output_tokens": int(summary["output_tokens"]),
        "replay_target_hits": int(summary["replay_target_hits"]),
        "throughput_tokens_per_s": float(summary["output_tokens_per_s"]),
        "ttft_p50_s": float(summary["ttft_p50_s"]),
        "ttft_p95_s": float(summary["ttft_p95_s"]),
        "tpot_p95_s": float(summary["tpot_p95_s"]),
        "max_itl_p95_s": float(summary["max_itl_p95_s"]),
        "max_itl_max_s": float(summary["max_itl_max_s"]),
        "e2e_p50_s": float(summary["e2e_p50_s"]),
        "e2e_p95_s": float(summary["e2e_p95_s"]),
        "e2e_mean_s": float(summary["e2e_mean_s"]),
        "scheduler": {
            key: _counter_delta(input_dir, prefix, metric) for key, metric in COUNTERS.items()
        },
    }


def _median_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    scalars = (
        "throughput_tokens_per_s",
        "ttft_p50_s",
        "ttft_p95_s",
        "tpot_p95_s",
        "max_itl_p95_s",
        "max_itl_max_s",
        "e2e_p50_s",
        "e2e_p95_s",
        "e2e_mean_s",
    )
    scheduler_keys = tuple(runs[0]["scheduler"])
    return {
        **{key: float(statistics.median(float(run[key]) for run in runs)) for key in scalars},
        "scheduler": {
            key: float(statistics.median(float(run["scheduler"][key]) for run in runs))
            for key in scheduler_keys
        },
    }


def _percent_change(before: float, after: float) -> float:
    return (after / before - 1.0) * 100.0


def _mismatch_count(left: dict[str, Any], right: dict[str, Any]) -> int:
    left_by_id = {
        str(result["request_id"]): result["generated_token_ids"] for result in left["results"]
    }
    right_by_id = {
        str(result["request_id"]): result["generated_token_ids"] for result in right["results"]
    }
    if left_by_id.keys() != right_by_id.keys():
        raise ValueError("request IDs differ across matched runs")
    return sum(left_by_id[key] != right_by_id[key] for key in left_by_id)


def _plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    median = summary["median"]
    runs = summary["runs"]
    labels = [LABELS[mode] for mode in MODES]
    colors = ("#355F8A", "#4AA3A2", "#C57B36", "#B4474E")
    x = np.arange(len(MODES))

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(16.5, 10.5))
    figure.suptitle(
        "Preemption trade-off on a 64-request production-length burst",
        fontsize=19,
    )

    throughput = [median[mode]["throughput_tokens_per_s"] for mode in MODES]
    bars = axes[0, 0].bar(x, throughput, color=colors, width=0.68)
    for index, mode in enumerate(MODES):
        axes[0, 0].scatter(
            [index] * 3,
            [run["throughput_tokens_per_s"] for run in runs[mode]],
            color="#202020",
            zorder=3,
            s=22,
        )
    axes[0, 0].bar_label(bars, fmt="%.1f", padding=3)
    axes[0, 0].set_ylabel("Output tokens/s")
    axes[0, 0].set_title("End-to-end throughput (higher is better)")
    axes[0, 0].set_xticks(x, labels, rotation=12, ha="right")

    width = 0.36
    p50 = [median[mode]["ttft_p50_s"] for mode in MODES]
    p95 = [median[mode]["ttft_p95_s"] for mode in MODES]
    axes[0, 1].bar(x - width / 2, p50, width, label="P50", color="#619CFF")
    axes[0, 1].bar(x + width / 2, p95, width, label="P95", color="#D89045")
    axes[0, 1].set_ylabel("Seconds")
    axes[0, 1].set_title("Time to first token (lower is better)")
    axes[0, 1].set_xticks(x, labels, rotation=12, ha="right")
    axes[0, 1].legend()

    gap_p95 = [median[mode]["max_itl_p95_s"] for mode in MODES]
    gap_max = [median[mode]["max_itl_max_s"] for mode in MODES]
    axes[1, 0].bar(x - width / 2, gap_p95, width, label="P95 request", color="#7C62A3")
    axes[1, 0].bar(x + width / 2, gap_max, width, label="Worst request", color="#CE6A85")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Seconds (log scale)")
    axes[1, 0].set_title("Largest in-stream token gap (lower is better)")
    axes[1, 0].set_xticks(x, labels, rotation=12, ha="right")
    axes[1, 0].legend()

    pauses = [median[mode]["scheduler"]["nonpreemptive_pauses"] for mode in MODES]
    handoffs = [median[mode]["scheduler"]["completion_claim_handoffs"] for mode in MODES]
    preemptions = [median[mode]["scheduler"]["vllm_preemptions"] for mode in MODES]
    axes[1, 1].bar(x - width, pauses, width, label="KV-retaining pauses", color="#4AA3A2")
    axes[1, 1].bar(x, handoffs, width, label="Claim handoffs", color="#355F8A")
    axes[1, 1].bar(x + width, preemptions, width, label="vLLM preemptions", color="#B4474E")
    axes[1, 1].set_ylabel("Events per run")
    axes[1, 1].set_title("Scheduler pressure events")
    axes[1, 1].set_xticks(x, labels, rotation=12, ha="right")
    axes[1, 1].legend()

    figure.text(
        0.5,
        0.01,
        "Same RTX 5090, Qwen2.5-Coder-7B BF16, 4096 KV tokens, 512-token batch budget, "
        "64 identical prompts and 7514 replayed output tokens; median of 3 post-warmup runs.",
        ha="center",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
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
            if (
                run["requests"] != 64
                or run["successful_requests"] != 64
                or run["failed_requests"] != 0
                or run["output_tokens"] != 7514
                or run["replay_target_hits"] != 64
            ):
                raise ValueError(f"unmatched or failed workload in {mode} r{run['run']}")

    median = {mode: _median_runs(mode_runs) for mode, mode_runs in runs.items()}
    strict = median["light-strict"]
    rolling = median["light-rolling10"]
    vllm_six = median["vllm-nopreempt6"]
    vllm_sixteen = median["vllm-preempt16"]

    reference_payload = _load(args.input_dir / "light-strict-r1.json")
    output_mismatches = {
        mode: _mismatch_count(
            reference_payload,
            _load(args.input_dir / f"{mode}-r1.json"),
        )
        for mode in MODES
    }

    summary = {
        "scope": {
            "device": "NVIDIA GeForce RTX 5090 32 GB",
            "model": "Qwen2.5-Coder-7B-Instruct",
            "dtype": "bfloat16",
            "block_size": 16,
            "kv_tokens": 4096,
            "max_num_scheduled_tokens": 512,
            "prefix_cache": False,
            "speculative_decoding": False,
            "ttft_admission": False,
            "workload": "64 ShareGPT first-turn prompts, burst arrival, production replay lengths",
            "prompt_tokens": sum(
                int(result["prompt_tokens"]) for result in reference_payload["results"]
            ),
            "output_tokens": 7514,
            "runs": "one full warmup then three measured runs per mode",
        },
        "runs": runs,
        "median": median,
        "paired_changes_percent": {
            "light_strict_to_rolling10": {
                "throughput": _percent_change(
                    strict["throughput_tokens_per_s"], rolling["throughput_tokens_per_s"]
                ),
                "ttft_p50": _percent_change(strict["ttft_p50_s"], rolling["ttft_p50_s"]),
                "ttft_p95": _percent_change(strict["ttft_p95_s"], rolling["ttft_p95_s"]),
                "max_itl_p95": _percent_change(strict["max_itl_p95_s"], rolling["max_itl_p95_s"]),
                "e2e_p95": _percent_change(strict["e2e_p95_s"], rolling["e2e_p95_s"]),
                "e2e_mean": _percent_change(strict["e2e_mean_s"], rolling["e2e_mean_s"]),
            },
            "vllm_nopreempt6_to_preempt16": {
                "throughput": _percent_change(
                    vllm_six["throughput_tokens_per_s"],
                    vllm_sixteen["throughput_tokens_per_s"],
                ),
                "ttft_p50": _percent_change(vllm_six["ttft_p50_s"], vllm_sixteen["ttft_p50_s"]),
                "ttft_p95": _percent_change(vllm_six["ttft_p95_s"], vllm_sixteen["ttft_p95_s"]),
                "max_itl_p95": _percent_change(
                    vllm_six["max_itl_p95_s"], vllm_sixteen["max_itl_p95_s"]
                ),
                "e2e_p95": _percent_change(vllm_six["e2e_p95_s"], vllm_sixteen["e2e_p95_s"]),
                "e2e_mean": _percent_change(vllm_six["e2e_mean_s"], vllm_sixteen["e2e_mean_s"]),
            },
            "cross_engine_nonpreemptive_gap": {
                "light_strict_vs_vllm_nopreempt6_throughput": _percent_change(
                    vllm_six["throughput_tokens_per_s"], strict["throughput_tokens_per_s"]
                )
            },
        },
        "output_reproducibility": {
            "r1_mismatches_against_light_strict": output_mismatches,
            "note": (
                "Prompts and replayed output lengths are identical, but greedy BF16 token IDs "
                "can diverge when scheduling changes batch shapes; token equality is not used as "
                "the performance gate."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(summary, args.figure)
    print(json.dumps(summary["paired_changes_percent"], indent=2))


if __name__ == "__main__":
    main()
