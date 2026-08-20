from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_preemption_comparison import COUNTERS, _counter_delta, _load

MODES = (
    "light-strict",
    "light-optimistic",
    "vllm-nopreempt6",
    "vllm-preempt16",
)
LABELS = {
    "light-strict": "Light strict\nTTFT SLO",
    "light-optimistic": "Light optimistic\nTTFT SLO",
    "vllm-nopreempt6": "vLLM no-preempt\n(6)",
    "vllm-preempt16": "vLLM default\n(16)",
}
COLORS = ("#355F8A", "#4AA3A2", "#C57B36", "#B4474E")
REQUESTS = 64
OFFERED_OUTPUT_TOKENS = 7514
TTFT_SLO_SECONDS = 1.25


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _run(input_dir: Path, mode: str, run: int) -> dict[str, Any]:
    prefix = f"{mode}-r{run}"
    payload = _load(input_dir / f"{prefix}.json")
    summary = payload["summary"]
    results = payload["results"]
    successful = [result for result in results if result["error"] is None]
    rejected = [result for result in results if result["status_code"] == 429]
    other_failures = [
        result for result in results if result["error"] is not None and result["status_code"] != 429
    ]
    ttfts = [float(result["ttft_s"]) for result in successful]
    tpot = [float(result["tpot_s"]) for result in successful if result["tpot_s"] is not None]
    max_gaps = [
        float(result["max_inter_token_gap_s"])
        for result in successful
        if result["max_inter_token_gap_s"] is not None
    ]
    e2e = [float(result["e2e_s"]) for result in successful]
    accepted_target_tokens = sum(int(result["target_output_tokens"]) for result in successful)
    output_tokens = int(summary["output_tokens"])
    if (
        int(summary["requests"]) != REQUESTS
        or len(results) != REQUESTS
        or int(summary["successful_requests"]) != len(successful)
        or int(summary["failed_requests"]) != len(rejected) + len(other_failures)
        or int(summary["replay_target_hits"]) != len(successful)
        or output_tokens != accepted_target_tokens
        or other_failures
    ):
        raise ValueError(f"invalid run: {input_dir.name}/{prefix}")
    if mode.startswith("light-"):
        if len(rejected) != int(summary["failed_requests"]):
            raise ValueError(f"non-429 light failure: {input_dir.name}/{prefix}")
    elif rejected or len(successful) != REQUESTS or output_tokens != OFFERED_OUTPUT_TOKENS:
        raise ValueError(f"unexpected vLLM rejection: {input_dir.name}/{prefix}")

    return {
        "run": run,
        "requests": REQUESTS,
        "successful_requests": len(successful),
        "rejected_429": len(rejected),
        "success_rate_percent": len(successful) / REQUESTS * 100.0,
        "output_tokens": output_tokens,
        "delivered_output_percent": output_tokens / OFFERED_OUTPUT_TOKENS * 100.0,
        "goodput_tokens_per_s": float(summary["output_tokens_per_s"]),
        "ttft_p50_s": _percentile(ttfts, 0.5),
        "ttft_p95_s": _percentile(ttfts, 0.95),
        "ttft_slo_compliance_percent": (
            sum(value <= TTFT_SLO_SECONDS for value in ttfts) / len(ttfts) * 100.0
        ),
        "tpot_p95_s": _percentile(tpot, 0.95),
        "max_itl_p95_s": _percentile(max_gaps, 0.95),
        "e2e_p95_s": _percentile(e2e, 0.95),
        "scheduler": {
            key: _counter_delta(input_dir, prefix, metric) for key, metric in COUNTERS.items()
        },
    }


def _median(runs: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = (
        "successful_requests",
        "rejected_429",
        "success_rate_percent",
        "output_tokens",
        "delivered_output_percent",
        "goodput_tokens_per_s",
        "ttft_p50_s",
        "ttft_p95_s",
        "ttft_slo_compliance_percent",
        "tpot_p95_s",
        "max_itl_p95_s",
        "e2e_p95_s",
    )
    scheduler_keys = tuple(runs[0]["scheduler"])
    return {
        **{key: float(statistics.median(float(run[key]) for run in runs)) for key in scalar_keys},
        "scheduler": {
            key: float(statistics.median(float(run["scheduler"][key]) for run in runs))
            for key in scheduler_keys
        },
    }


def _scenario(input_dir: Path, request_rate: float) -> dict[str, Any]:
    runs = {mode: [_run(input_dir, mode, run) for run in range(1, 4)] for mode in MODES}
    for mode in MODES:
        for run in range(1, 4):
            metadata = _load(input_dir / f"{mode}-r{run}.json")["metadata"]
            if (
                metadata["arrival_mode"] != "poisson"
                or float(metadata["request_rate"]) != request_rate
            ):
                raise ValueError(f"unmatched arrival metadata: {input_dir.name}/{mode}-r{run}")
    return {
        "request_rate": request_rate,
        "runs": runs,
        "median": {mode: _median(mode_runs) for mode, mode_runs in runs.items()},
    }


def _plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 6, figsize=(26, 9.5))
    figure.suptitle("Predictive TTFT admission at 1.25 s: service trade-offs", fontsize=20)
    x = np.arange(len(MODES))
    for row, (scenario_name, scenario) in enumerate(summary["scenarios"].items()):
        median = scenario["median"]
        title = f"{scenario_name}: {scenario['request_rate']:.0f} req/s"
        metrics = (
            ("success_rate_percent", "Request success (%)", "%.1f"),
            ("delivered_output_percent", "Delivered output (%)", "%.1f"),
            ("goodput_tokens_per_s", "Output goodput (tok/s)", "%.0f"),
            ("ttft_p95_s", "Accepted TTFT P95 (s)", "%.3g"),
            ("max_itl_p95_s", "Accepted max gap P95 (s)", "%.3g"),
        )
        for column, (key, label, fmt) in enumerate(metrics):
            values = [median[mode][key] for mode in MODES]
            bars = axes[row, column].bar(x, values, color=COLORS)
            axes[row, column].bar_label(bars, fmt=fmt, padding=3, fontsize=8)
            axes[row, column].set_title(f"{title}\n{label}")
            axes[row, column].set_ylabel(label)
            axes[row, column].set_xticks(x, [LABELS[mode] for mode in MODES], rotation=12)
            if key == "ttft_p95_s":
                axes[row, column].axhline(
                    TTFT_SLO_SECONDS,
                    color="#222222",
                    linestyle="--",
                    linewidth=1.2,
                    label="1.25 s target",
                )
                axes[row, column].legend(fontsize=8)

        rejected = [median[mode]["rejected_429"] for mode in MODES]
        resubmits = [median[mode]["scheduler"]["self_resubmits"] for mode in MODES]
        preemptions = [median[mode]["scheduler"]["vllm_preemptions"] for mode in MODES]
        width = 0.25
        axes[row, 5].bar(x - width, rejected, width, label="HTTP 429", color="#7B61A8")
        axes[row, 5].bar(x, resubmits, width, label="Self-resubmit", color="#4AA3A2")
        axes[row, 5].bar(x + width, preemptions, width, label="vLLM preempt", color="#B4474E")
        axes[row, 5].set_title(f"{title}\nControl events/run")
        axes[row, 5].set_ylabel("Events/run")
        axes[row, 5].set_xticks(x, [LABELS[mode] for mode in MODES], rotation=12)
        axes[row, 5].legend(fontsize=8)

    figure.text(
        0.5,
        0.01,
        "RTX 5090 · Qwen2.5-Coder-7B BF16 · 4096 KV tokens · Poisson traffic · "
        "64 identical prompts · median of 3 post-warmup runs · vLLM has no admission SLO",
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
            "light_ttft_slo_seconds": TTFT_SLO_SECONDS,
            "light_other_ttft_gates": "max pending off; KV watermark off",
            "vllm_ttft_admission": False,
            "requests": REQUESTS,
            "offered_output_tokens": OFFERED_OUTPUT_TOKENS,
            "runs": "one warmup then three measured runs per mode and rate",
        },
        "scenarios": {
            "steady": _scenario(args.steady_dir, 2.0),
            "pressure": _scenario(args.pressure_dir, 8.0),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(summary, args.figure)
    print(
        json.dumps(
            {name: value["median"] for name, value in summary["scenarios"].items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
