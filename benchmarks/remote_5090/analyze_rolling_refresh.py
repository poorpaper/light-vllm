from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class Case:
    workload: str
    label: str
    files: tuple[str, ...]
    metrics: bool = False


CASES = (
    Case(
        "Steady 4 req/s",
        "Rolling 10%\nold",
        tuple(f"final10-light-rolling10-production-replay-r{run}.json" for run in range(1, 4)),
        metrics=True,
    ),
    Case(
        "Steady 4 req/s",
        "Rolling 10%\neager refresh",
        tuple(f"final15-light-rolling10-refresh-steady-r{run}.json" for run in range(1, 4)),
        metrics=True,
    ),
    Case(
        "Steady 4 req/s",
        "vLLM",
        tuple(f"final9-vllm-production-replay-r{run}.json" for run in range(1, 4)),
    ),
    Case(
        "64-request burst",
        "Rolling 10%\nold",
        tuple(f"final11-light-rolling10-production-burst-r{run}.json" for run in range(1, 4)),
        metrics=True,
    ),
    Case(
        "64-request burst",
        "Rolling 10%\neager refresh",
        tuple(f"final16-light-rolling10-refresh-burst-r{run}.json" for run in range(1, 4)),
        metrics=True,
    ),
    Case(
        "64-request burst",
        "vLLM",
        tuple(f"final11-vllm-production-burst-r{run}.json" for run in range(1, 4)),
    ),
)

COUNTERS = {
    "nonpreemptive_pauses": "light_vllm_nonpreemptive_pauses_total",
    "completion_claim_handoffs": "light_vllm_completion_claim_handoffs_total",
    "self_resubmits": "light_vllm_self_resubmits_total",
    "rolled_back_tokens": "light_vllm_rolled_back_tokens_total",
}


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _counter_value(path: Path, metric: str) -> float:
    pattern = re.compile(rf"^{re.escape(metric)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match is not None:
            return float(match.group(1))
    return 0.0


def _load_run(raw_dir: Path, filename: str, with_metrics: bool) -> dict[str, float]:
    path = raw_dir / filename
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
    run = {
        "output_tokens_per_s": float(summary["output_tokens_per_s"]),
        "ttft_p50_s": float(summary["ttft_p50_s"]),
        "ttft_p95_s": float(summary["ttft_p95_s"]),
        "max_inter_token_gap_s": float(summary["max_itl_max_s"]),
        "output_tokens": float(summary["output_tokens"]),
    }
    if with_metrics:
        stem = path.with_suffix("")
        before = Path(f"{stem}.metrics-before.prom")
        after = Path(f"{stem}.metrics-after.prom")
        run.update(
            {
                name: _counter_value(after, metric) - _counter_value(before, metric)
                for name, metric in COUNTERS.items()
            }
        )
    return run


def _summarize_case(raw_dir: Path, case: Case) -> dict[str, Any]:
    runs = [_load_run(raw_dir, filename, case.metrics) for filename in case.files]
    metric_names = tuple(runs[0])
    aggregate = {
        name: {
            "mean": _mean([run[name] for run in runs]),
            "stdev": _stdev([run[name] for run in runs]),
        }
        for name in metric_names
    }
    return {
        "workload": case.workload,
        "label": case.label.replace("\n", " "),
        "files": list(case.files),
        "runs": runs,
        "aggregate": aggregate,
    }


def _plot(summary: list[dict[str, Any]], output: Path) -> None:
    workloads = ("Steady 4 req/s", "64-request burst")
    metrics = (
        ("output_tokens_per_s", "Output throughput (tok/s)"),
        ("ttft_p95_s", "TTFT P95 (s)"),
        ("max_inter_token_gap_s", "Worst token gap (s)"),
    )
    colors = ("#9D755D", "#4C78A8", "#54A24B")
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
    for row, workload in enumerate(workloads):
        cases = [case for case in summary if case["workload"] == workload]
        labels = [case["label"].replace(" ", "\n", 2) for case in cases]
        for column, (metric, title) in enumerate(metrics):
            axis = axes[row][column]
            means = [case["aggregate"][metric]["mean"] for case in cases]
            errors = [case["aggregate"][metric]["stdev"] for case in cases]
            bars = axis.bar(
                range(len(cases)),
                means,
                yerr=errors,
                capsize=4,
                color=colors,
                alpha=0.9,
            )
            axis.set_xticks(range(len(cases)), labels, fontsize=9)
            axis.set_title(f"{workload}: {title}", fontsize=11)
            axis.grid(axis="y", alpha=0.25)
            for bar, value in zip(bars, means, strict=True):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    figure.suptitle(
        "Rolling-claim eager refresh A/B; error bars are 3-run sample SD",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    raw_dir = args.results_dir / "raw"
    summary = [_summarize_case(raw_dir, case) for case in CASES]
    output = {
        "notes": [
            "All cases use the same 64 ShareGPT prompts and deliver 7514 output tokens per run.",
            "The eager-refresh change keeps request KV and does not enable self-resubmit.",
            "Scheduler counters are per-run deltas from Prometheus snapshots.",
        ],
        "cases": summary,
    }
    (args.results_dir / "rolling_refresh_summary.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    _plot(summary, args.results_dir / "figures" / "07_rolling_claim_refresh.png")


if __name__ == "__main__":
    main()
