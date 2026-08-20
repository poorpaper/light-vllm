from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

TTFT_SLO_SECONDS = 1.25


@dataclass(frozen=True, slots=True)
class Case:
    workload: str
    label: str
    files: tuple[str, ...]


CASES = (
    Case(
        "Steady 4 req/s",
        "Rolling 10%\nno admission",
        tuple(f"final10-light-rolling10-production-replay-r{run}.json" for run in range(1, 4)),
    ),
    Case(
        "Steady 4 req/s",
        "Rolling 10%\nTTFT = 1.25s",
        tuple(
            f"final14-light-rolling10-ttft125-calibrated-steady-r{run}.json" for run in range(1, 4)
        ),
    ),
    Case(
        "Steady 4 req/s",
        "vLLM",
        tuple(f"final9-vllm-production-replay-r{run}.json" for run in range(1, 4)),
    ),
    Case(
        "64-request burst",
        "Rolling 10%\nno admission",
        tuple(f"final11-light-rolling10-production-burst-r{run}.json" for run in range(1, 4)),
    ),
    Case(
        "64-request burst",
        "Rolling 10%\nTTFT = 1.25s",
        tuple(
            f"final13-light-rolling10-ttft125-calibrated-burst-r{run}.json" for run in range(1, 4)
        ),
    ),
    Case(
        "64-request burst",
        "vLLM",
        tuple(f"final11-vllm-production-burst-r{run}.json" for run in range(1, 4)),
    ),
)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _load_run(path: Path) -> dict[str, float | int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    successful_results = [result for result in payload["results"] if result["error"] is None]
    successful_ttft = [
        float(result["ttft_s"]) for result in successful_results if result["ttft_s"] is not None
    ]
    overloaded = sum(result["status_code"] == 429 for result in payload["results"])
    requests = int(summary["requests"])
    successful = int(summary["successful_requests"])
    return {
        "requests": requests,
        "successful_requests": successful,
        "overloaded_requests": overloaded,
        "admission_rate": successful / requests,
        "offered_output_tokens_per_s": float(summary["output_tokens_per_s"]),
        "accepted_ttft_p50_s": float(summary["ttft_p50_s"]),
        "accepted_ttft_p95_s": float(summary["ttft_p95_s"]),
        "accepted_slo_attainment": (
            sum(value <= TTFT_SLO_SECONDS for value in successful_ttft) / len(successful_ttft)
        ),
        "max_inter_token_gap_s": float(summary["max_itl_max_s"]),
        "delivered_output_tokens": int(summary["output_tokens"]),
    }


def _summarize_case(raw_dir: Path, case: Case) -> dict[str, Any]:
    runs = [_load_run(raw_dir / filename) for filename in case.files]
    metric_names = (
        "admission_rate",
        "offered_output_tokens_per_s",
        "accepted_ttft_p50_s",
        "accepted_ttft_p95_s",
        "accepted_slo_attainment",
        "max_inter_token_gap_s",
        "delivered_output_tokens",
        "overloaded_requests",
    )
    aggregate = {
        name: {
            "mean": _mean([float(run[name]) for run in runs]),
            "stdev": _stdev([float(run[name]) for run in runs]),
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
        ("admission_rate", "Requests admitted (%)", 100.0),
        ("offered_output_tokens_per_s", "Delivered output (tok/s)", 1.0),
        ("accepted_ttft_p95_s", "Accepted-request TTFT P95 (s)", 1.0),
    )
    colors = ("#4C78A8", "#F58518", "#54A24B")
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
    for row, workload in enumerate(workloads):
        cases = [case for case in summary if case["workload"] == workload]
        labels = [case["label"].replace(" ", "\n", 2) for case in cases]
        for column, (metric, title, scale) in enumerate(metrics):
            axis = axes[row][column]
            means = [case["aggregate"][metric]["mean"] * scale for case in cases]
            errors = [case["aggregate"][metric]["stdev"] * scale for case in cases]
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
            if metric == "admission_rate":
                axis.set_ylim(0, 108)
            if metric == "accepted_ttft_p95_s":
                axis.axhline(
                    TTFT_SLO_SECONDS,
                    color="#E45756",
                    linestyle="--",
                    linewidth=1.5,
                    label="1.25s threshold",
                )
                axis.legend(fontsize=8)
            for bar, value in zip(bars, means, strict=True):
                text = f"{value:.0f}%" if metric == "admission_rate" else f"{value:.2f}"
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    text,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    figure.suptitle(
        "TTFT admission trades accepted load for latency; error bars are 3-run sample SD",
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
        "ttft_slo_seconds": TTFT_SLO_SECONDS,
        "notes": [
            "TTFT and SLO attainment include only admitted requests.",
            (
                "Offered output throughput counts only delivered tokens, so admission rate "
                "must be read alongside it."
            ),
            (
                "The TTFT=1.25s cases use 100 minimum observations and a matched-workload "
                "calibration run."
            ),
        ],
        "cases": summary,
    }
    (args.results_dir / "ttft_admission_summary.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    _plot(summary, args.results_dir / "figures" / "06_ttft_admission.png")


if __name__ == "__main__":
    main()
