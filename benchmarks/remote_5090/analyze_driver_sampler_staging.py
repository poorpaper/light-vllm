from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

RATES = (2, 4, 6, 8)
MODES = ("light-strict", "vllm-eager", "vllm-default")
MODE_LABELS = {
    "light-strict": "Light strict",
    "vllm-eager": "vLLM eager",
    "vllm-default": "vLLM default",
}
MODE_COLORS = {
    "light-strict": "#E45756",
    "vllm-eager": "#4C78A8",
    "vllm-default": "#54A24B",
}
QUANTILES = (0.5, 0.9, 0.95, 0.99)
SLO_MS = (50, 100, 200)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _median(values: Iterable[float]) -> float:
    return statistics.median(values)


def _formal_runs(matrix_root: Path, rate: int, mode: str) -> list[dict[str, Any]]:
    paths = [matrix_root / f"rate{rate}" / f"{mode}-normal-loaded-r{run}.json" for run in (1, 2, 3)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing formal runs: {missing}")
    return [_load(path) for path in paths]


def _successful_requests(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        request
        for request in run["results"]
        if request.get("status_code") == 200 and request.get("ttft_s") is not None
    ]


def _summarize_mode(rate: int, mode: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    pooled = [request for run in runs for request in _successful_requests(run)]
    ttft_ms = [float(request["ttft_s"]) * 1000 for request in pooled]
    throughput_runs = [float(run["summary"]["output_tokens_per_s"]) for run in runs]
    elapsed_runs = [float(run["summary"]["elapsed_s"]) for run in runs]
    successful_runs = [int(run["summary"]["successful_requests"]) for run in runs]
    requested_runs = [int(run["summary"]["requests"]) for run in runs]
    goodput_runs: dict[str, list[float]] = {str(slo): [] for slo in SLO_MS}
    for run, elapsed_s in zip(runs, elapsed_runs, strict=True):
        requests = _successful_requests(run)
        for slo in SLO_MS:
            eligible_tokens = sum(
                int(request["output_tokens"])
                for request in requests
                if float(request["ttft_s"]) * 1000 <= slo
            )
            goodput_runs[str(slo)].append(eligible_tokens / elapsed_s)

    quantiles = {
        f"p{round(quantile * 100):02d}": _percentile(ttft_ms, quantile) for quantile in QUANTILES
    }
    return {
        "rate_req_s": rate,
        "mode": mode,
        "formal_runs": len(runs),
        "pooled_successful_requests": len(pooled),
        "successful_requests_per_run": successful_runs,
        "requested_requests_per_run": requested_runs,
        "all_requests_successful": successful_runs == requested_runs,
        "throughput_tok_s_runs": throughput_runs,
        "throughput_tok_s_median": _median(throughput_runs),
        "ttft_ms": quantiles,
        "slo_goodput_tok_s_median": {slo: _median(values) for slo, values in goodput_runs.items()},
    }


def _read_matrix(input_root: Path) -> list[dict[str, Any]]:
    matrix_root = input_root / "normal-matrix"
    rows: list[dict[str, Any]] = []
    for rate in RATES:
        for mode in MODES:
            rows.append(_summarize_mode(rate, mode, _formal_runs(matrix_root, rate, mode)))

    eager_by_rate = {
        int(row["rate_req_s"]): float(row["throughput_tok_s_median"])
        for row in rows
        if row["mode"] == "vllm-eager"
    }
    for row in rows:
        eager = eager_by_rate[int(row["rate_req_s"])]
        row["throughput_vs_vllm_eager"] = float(row["throughput_tok_s_median"]) / eager
    return rows


def _find_profile_json(directory: Path) -> Path | None:
    paths = sorted(directory.glob("*-profile.*.json"))
    return paths[0] if paths else None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _light_profile(input_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = (
        input_root / "final-profiles" / "light" / "light-strict-normal-loaded-profile-stages.json"
    )
    if not path.is_file():
        return [], {"available": False, "reason": f"missing {path}"}
    payload = _load(path)
    groups = {
        "Light decode W1": [step for step in payload["steps"] if step["query_width"] == 1],
        "Light mixed": [
            step for step in payload["steps"] if step["query_width"] > 1 and step["batch_size"] > 1
        ],
    }
    rows: list[dict[str, Any]] = []
    for label, steps in groups.items():
        executor = [float(step["executor_wall_ms"]) for step in steps]
        cuda = [float(step["cuda_event_ms"]) for step in steps]
        gaps = [
            float(step["previous_executor_gap_ms"])
            for step in steps
            if step["previous_executor_gap_ms"] is not None
        ]
        gap = _mean(gaps)
        executor_mean = _mean(executor)
        rows.append(
            {
                "label": label,
                "kind": "light",
                "steps": len(steps),
                "service_ms": (
                    executor_mean + gap if executor_mean is not None and gap is not None else None
                ),
                "executor_ms": executor_mean,
                "cuda_ms": _mean(cuda),
                "gap_ms": gap,
            }
        )
    return rows, {
        "available": True,
        "path": str(path),
        "definition": (
            "Light service is mean executor wall plus the mean gap immediately before the "
            "classified step. Mixed means query_width > 1 and batch_size > 1."
        ),
    }


def _vllm_profiles(input_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    for mode in ("vllm-eager", "vllm-default"):
        path = _find_profile_json(input_root / "final-profiles" / mode)
        if path is None:
            sources[mode] = {"available": False, "reason": "profile JSON not found"}
            continue
        payload = _load(path)
        steps = int(payload["engine_steps"])
        gap_divisor = max(steps - 1, 1)
        execute_stage = payload["stages"].get("executor.execute_model", {})
        rows.append(
            {
                "label": MODE_LABELS[mode] + " overall",
                "kind": "vllm",
                "steps": steps,
                "service_ms": float(payload["engine_step_span_ms"]) / steps,
                "executor_ms": execute_stage.get("wall_mean_ms"),
                "cuda_ms": None,
                "gap_ms": float(payload["inter_step_gap_total_ms"]) / gap_divisor,
            }
        )
        sources[mode] = {"available": True, "path": str(path)}
    sources["limitation"] = (
        "The vLLM hook reports aggregate engine/executor wall time only. It does not record "
        "per-step query shape or a CUDA-event duration, so mixed/decode and CUDA are unavailable."
    )
    return rows, sources


def _read_profiles(input_root: Path) -> dict[str, Any]:
    light_rows, light_source = _light_profile(input_root)
    vllm_rows, vllm_sources = _vllm_profiles(input_root)
    return {
        "rows": light_rows + vllm_rows,
        "sources": {"light": light_source, "vllm": vllm_sources},
    }


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    fieldnames = [
        "rate_req_s",
        "mode",
        "formal_runs",
        "pooled_successful_requests",
        "all_requests_successful",
        "throughput_tok_s_median",
        "throughput_vs_vllm_eager",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "goodput_ttft_le_50ms_tok_s",
        "goodput_ttft_le_100ms_tok_s",
        "goodput_ttft_le_200ms_tok_s",
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "rate_req_s": row["rate_req_s"],
                    "mode": row["mode"],
                    "formal_runs": row["formal_runs"],
                    "pooled_successful_requests": row["pooled_successful_requests"],
                    "all_requests_successful": row["all_requests_successful"],
                    "throughput_tok_s_median": row["throughput_tok_s_median"],
                    "throughput_vs_vllm_eager": row["throughput_vs_vllm_eager"],
                    **{f"ttft_{key}_ms": value for key, value in row["ttft_ms"].items()},
                    **{
                        f"goodput_ttft_le_{slo}ms_tok_s": value
                        for slo, value in row["slo_goodput_tok_s_median"].items()
                    },
                }
            )


def _plot_ttft(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8), sharex=True, constrained_layout=True)
    width = 0.24
    x = np.arange(len(RATES))
    for axis, quantile in zip(axes.flat, ("p50", "p90", "p95", "p99"), strict=True):
        for index, mode in enumerate(MODES):
            values = [
                next(
                    row["ttft_ms"][quantile]
                    for row in rows
                    if row["rate_req_s"] == rate and row["mode"] == mode
                )
                for rate in RATES
            ]
            axis.bar(
                x + (index - 1) * width,
                values,
                width,
                color=MODE_COLORS[mode],
                label=MODE_LABELS[mode],
            )
        axis.set_title(f"TTFT {quantile.upper()}")
        axis.set_ylabel("Milliseconds")
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(x, [str(rate) for rate in RATES])
    for axis in axes[-1]:
        axis.set_xlabel("Poisson request rate (req/s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3)
    figure.suptitle("Normal-load TTFT distribution (3 runs, 192 requests per point)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_throughput(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for mode in MODES:
        selected = [
            next(row for row in rows if row["mode"] == mode and row["rate_req_s"] == rate)
            for rate in RATES
        ]
        throughputs = [row["throughput_tok_s_median"] for row in selected]
        ratios = [row["throughput_vs_vllm_eager"] * 100 for row in selected]
        axes[0].plot(
            RATES,
            throughputs,
            marker="o",
            linewidth=2,
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
        )
        axes[1].plot(
            RATES,
            ratios,
            marker="o",
            linewidth=2,
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
        )
    axes[0].set_title("Output throughput")
    axes[0].set_ylabel("Output tokens/s (median of 3 runs)")
    axes[1].set_title("Throughput relative to vLLM eager")
    axes[1].set_ylabel("Percent")
    axes[1].axhline(85, color="#777777", linestyle="--", linewidth=1, label="85% target")
    for axis in axes:
        axis.set_xlabel("Poisson request rate (req/s)")
        axis.set_xticks(RATES)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    figure.suptitle("Matched normal-load throughput; rate 2 is arrival-limited")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_profiles(profile: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = profile["rows"]
    if not rows:
        return
    labels = [row["label"] for row in rows]
    metrics = (
        ("service_ms", "Service"),
        ("executor_ms", "Executor/model hook"),
        ("cuda_ms", "CUDA event"),
        ("gap_ms", "Step gap"),
    )
    x = np.arange(len(rows))
    width = 0.19
    figure, axis = plt.subplots(figsize=(12, 5.4), constrained_layout=True)
    for index, (key, label) in enumerate(metrics):
        values = [float(row[key]) if row[key] is not None else np.nan for row in rows]
        bars = axis.bar(x + (index - 1.5) * width, values, width, label=label)
        if key == "cuda_ms":
            for row_index, value in enumerate(values):
                if np.isnan(value):
                    axis.text(
                        x[row_index] + (index - 1.5) * width,
                        0.15,
                        "N/A",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=7,
                    )
        axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Mean milliseconds per measured step")
    axis.set_title(
        "Final rate-8 profile: comparable boundaries and known limitations\n"
        "vLLM has no per-shape split or CUDA event; N/A is not zero."
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_goodput(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(SLO_MS), figsize=(14.5, 4.5), constrained_layout=True)
    for axis, slo in zip(axes, SLO_MS, strict=True):
        for mode in MODES:
            values = [
                next(
                    float(row["slo_goodput_tok_s_median"][str(slo)])
                    for row in rows
                    if row["rate_req_s"] == rate and row["mode"] == mode
                )
                for rate in RATES
            ]
            axis.plot(
                RATES,
                values,
                marker="o",
                color=MODE_COLORS[mode],
                label=MODE_LABELS[mode],
            )
        axis.set_title(f"TTFT <= {slo} ms")
        axis.set_xlabel("Poisson request rate (req/s)")
        axis.set_xticks(RATES)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("SLO-qualified output tokens/s")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3)
    figure.suptitle("TTFT SLO goodput (all thresholds reported, median of 3 runs)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _validate_name_contract(matrix_root: Path) -> None:
    formal = re.compile(r"-(?:r1|r2|r3)\.json$")
    selected = [path for path in matrix_root.rglob("*.json") if formal.search(path.name)]
    expected = len(RATES) * len(MODES) * 3
    if len(selected) != expected:
        raise ValueError(f"expected {expected} formal JSON files, found {len(selected)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the final driver/sampler/staging experiment without warmups."
    )
    parser.add_argument(
        "input_root",
        nargs="?",
        type=Path,
        default=Path("benchmarks/remote_5090/results/2026-08-21-driver-sampler-staging"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_dir = (args.output_dir or input_root / "analysis").resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _validate_name_contract(input_root / "normal-matrix")

    matrix_rows = _read_matrix(input_root)
    profiles = _read_profiles(input_root)
    summary = {
        "scope": {
            "input_root": str(input_root),
            "rates_req_s": list(RATES),
            "modes": list(MODES),
            "formal_runs_per_point": 3,
            "excluded": ["warmup", "profile"],
            "aggregation": (
                "Throughput and SLO goodput are medians of three formal runs. TTFT "
                "quantiles pool the 192 successful requests from those runs."
            ),
        },
        "matrix": matrix_rows,
        "profiles": profiles,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(matrix_rows, output_dir / "summary.csv")
    _plot_ttft(matrix_rows, figures_dir / "01_ttft_distribution.png")
    _plot_throughput(matrix_rows, figures_dir / "02_throughput_vs_eager.png")
    _plot_profiles(profiles, figures_dir / "03_profile_breakdown.png")
    _plot_goodput(matrix_rows, figures_dir / "04_ttft_slo_goodput.png")
    print(output_dir)


if __name__ == "__main__":
    main()
