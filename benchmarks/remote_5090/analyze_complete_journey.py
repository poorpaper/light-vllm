from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

FINAL_LIGHT = "#E45756"
VLLM_EAGER = "#4C78A8"
NO_SPEC = "#9D9D9D"
CHAIN = "#F2CF5B"
TRIE = "#59A14F"


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], percentile: float) -> float:
    """使用线性插值重算单轮请求百分位，与常见 benchmark 口径一致。"""
    if not values:
        raise ValueError("cannot calculate a percentile from an empty sample")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _successful_results(run: dict[str, Any]) -> list[dict[str, Any]]:
    results = [result for result in run["results"] if int(result["status_code"]) == 200]
    summary = run["summary"]
    if len(results) != int(summary["requests"]):
        raise ValueError("formal run contains failed or missing requests")
    if len(results) != int(summary["successful_requests"]):
        raise ValueError("successful request count does not match the result rows")
    return results


def _summarize_runs(paths: list[Path]) -> dict[str, Any]:
    per_run: list[dict[str, float]] = []
    for path in paths:
        run = _load(path)
        results = _successful_results(run)
        ttft_ms = [float(result["ttft_s"]) * 1000 for result in results]
        tpot_ms = [float(result["tpot_s"]) * 1000 for result in results]
        per_run.append(
            {
                "throughput_tok_s": float(run["summary"]["output_tokens_per_s"]),
                **{
                    f"ttft_p{percentile}_ms": _percentile(ttft_ms, percentile)
                    for percentile in (50, 90, 95)
                },
                **{
                    f"tpot_p{percentile}_ms": _percentile(tpot_ms, percentile)
                    for percentile in (50, 90, 95)
                },
            }
        )
    keys = tuple(per_run[0])
    return {
        "runs": len(per_run),
        "requests_per_run": 64,
        "aggregation": "median of per-run metrics",
        "per_run": per_run,
        **{key: statistics.median(row[key] for row in per_run) for key in keys},
    }


def _light_paths(root: Path, case: str) -> list[Path]:
    return [root / f"candidate-r{index}" / f"light-strict-{case}-r1.json" for index in (1, 2, 3)]


def _vllm_paths(root: Path, case: str) -> list[Path]:
    return [root / f"vllm-eager-{case}-r{index}.json" for index in (1, 2, 3)]


def _prometheus_value(path: Path, metric: str) -> float:
    prefix = metric + "{"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return float(line.rsplit(" ", 1)[1])
    raise KeyError(f"{metric} is absent from {path}")


def _speculation_observation(json_path: Path) -> dict[str, float]:
    before = json_path.with_suffix(".metrics-before.prom")
    after = json_path.with_suffix(".metrics-after.prom")
    names = {
        "attempts": "light_vllm_speculation_attempts_total",
        "proposed": "light_vllm_speculative_proposed_nodes_total",
        "accepted": "light_vllm_speculative_accepted_nodes_total",
        "verified": "light_vllm_speculative_verified_tokens_total",
    }
    deltas = {
        name: _prometheus_value(after, metric) - _prometheus_value(before, metric)
        for name, metric in names.items()
    }
    if deltas["attempts"] <= 0:
        raise ValueError(f"no speculation attempts in {json_path}")
    return {
        "mean_proposed_nodes": deltas["proposed"] / deltas["attempts"],
        "mean_accepted_draft_tokens": deltas["accepted"] / deltas["attempts"],
        "mean_verified_tokens": deltas["verified"] / deltas["attempts"],
        "draft_acceptance_ratio": deltas["accepted"] / deltas["proposed"],
    }


def _summarize_speculation(paths: list[Path], *, has_speculation: bool) -> dict[str, Any]:
    runs = [_load(path) for path in paths]
    output: dict[str, Any] = {
        "runs": len(runs),
        "throughput_tok_s": statistics.median(
            float(run["summary"]["output_tokens_per_s"]) for run in runs
        ),
        "ttft_p50_ms": statistics.median(
            float(run["summary"]["ttft_p50_s"]) * 1000 for run in runs
        ),
        "ttft_p95_ms": statistics.median(
            float(run["summary"]["ttft_p95_s"]) * 1000 for run in runs
        ),
        "tpot_p50_ms": statistics.median(
            float(run["summary"]["tpot_p50_s"]) * 1000 for run in runs
        ),
        "tpot_p95_ms": statistics.median(
            float(run["summary"]["tpot_p95_s"]) * 1000 for run in runs
        ),
    }
    if has_speculation:
        observations = [_speculation_observation(path) for path in paths]
        for key in observations[0]:
            output[key] = statistics.median(row[key] for row in observations)
    return output


def _final_results(results_root: Path) -> dict[str, Any]:
    fairness = results_root / "2026-08-22-driver-control-fairness"
    loaded_light = fairness / "driver-control-fairness-ab-192d050-20260822"
    low_light = fairness / "driver-control-fairness-low-ab-192d050-20260822"
    loaded_vllm = fairness / "driver-control-fairness-vllm-eager-20260822"
    low_vllm = results_root / "2026-08-22-final90-hotpath" / "final90-vllm-eager-low-20260822"
    cases: dict[str, Any] = {}
    for case, light_root, vllm_root in (
        ("normal-low", low_light, low_vllm),
        ("normal-loaded", loaded_light, loaded_vllm),
    ):
        light = _summarize_runs(_light_paths(light_root, case))
        eager = _summarize_runs(_vllm_paths(vllm_root, case))
        cases[case] = {
            "light": light,
            "vllm_eager": eager,
            "ratios": {
                "throughput_percent": 100 * light["throughput_tok_s"] / eager["throughput_tok_s"],
                "tpot_p50_speed_percent": 100 * eager["tpot_p50_ms"] / light["tpot_p50_ms"],
                "ttft_p95_ratio": light["ttft_p95_ms"] / eager["ttft_p95_ms"],
            },
        }
    return cases


def _speculation_results(results_root: Path) -> dict[str, Any]:
    formal = results_root / "2026-08-19-qwen25-coder-7b" / "raw"
    rescue = results_root / "2026-08-20-profiled-qwen25-coder-7b" / "raw"
    return {
        "repetition_code": {
            "scope": "stable formal runs after the separate warmup",
            "no_spec": _summarize_speculation(
                [formal / f"formal-light-7b-spec-nospec-r{index}.json" for index in (2, 3, 4)],
                has_speculation=False,
            ),
            "chain_7": _summarize_speculation(
                [formal / f"formal-light-7b-chain7-r{index}.json" for index in (2, 3, 4)],
                has_speculation=True,
            ),
            "trie_7": _summarize_speculation(
                [formal / f"formal-light-7b-trie7-r{index}.json" for index in (1, 2, 3)],
                has_speculation=True,
            ),
        },
        "branch_rescue": {
            "scope": "selected decoy-prefix workload designed to exercise trie branches",
            "no_spec": _summarize_speculation(
                [rescue / f"final-light-rescue-nospec-r{index}.json" for index in (1, 2)],
                has_speculation=False,
            ),
            "chain_7": _summarize_speculation(
                [rescue / f"final-light-rescue-chain7-r{index}.json" for index in (1, 2)],
                has_speculation=True,
            ),
            "trie_7": _summarize_speculation(
                [rescue / f"final-light-rescue-trie7-r{index}.json" for index in (1, 2)],
                has_speculation=True,
            ),
        },
    }


def _plot_final_throughput(cases: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ("2 req/s", "8 req/s")
    case_names = ("normal-low", "normal-loaded")
    x = np.arange(len(labels))
    width = 0.32
    figure, axis = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    for offset, system, label, color in (
        (-width / 2, "light", "Light strict (final)", FINAL_LIGHT),
        (width / 2, "vllm_eager", "vLLM eager", VLLM_EAGER),
    ):
        values = [cases[case][system]["throughput_tok_s"] for case in case_names]
        bars = axis.bar(x + offset, values, width, label=label, color=color)
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    axis.set_title("Final output throughput (median of 3 runs)")
    axis.set_ylabel("output tokens/s")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_final_latencies(cases: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    percentiles = (50, 90, 95)
    x = np.arange(len(percentiles))
    width = 0.34
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    for row, (case, rate) in enumerate((("normal-low", 2), ("normal-loaded", 8))):
        for column, (metric, title, unit) in enumerate(
            (("ttft", "TTFT", "ms/request"), ("tpot", "TPOT", "ms/token"))
        ):
            axis = axes[row, column]
            for offset, system, label, color in (
                (-width / 2, "light", "Light strict (final)", FINAL_LIGHT),
                (width / 2, "vllm_eager", "vLLM eager", VLLM_EAGER),
            ):
                values = [
                    cases[case][system][f"{metric}_p{percentile}_ms"] for percentile in percentiles
                ]
                bars = axis.bar(x + offset, values, width, label=label, color=color)
                axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
            axis.set_title(f"{rate} req/s {title}")
            axis.set_ylabel(unit)
            axis.set_xticks(x, [f"P{percentile}" for percentile in percentiles])
            axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2)
    figure.suptitle("Final latency percentiles (median of per-run percentiles)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_speculation(results: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    modes = ("no_spec", "chain_7", "trie_7")
    labels = ("No spec", "Chain-7", "Trie-7")
    colors = (NO_SPEC, CHAIN, TRIE)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for axis, workload, title in zip(
        axes,
        ("repetition_code", "branch_rescue"),
        ("Repetition-code workload", "Selected branch-rescue workload"),
        strict=True,
    ):
        values = [results[workload][mode]["throughput_tok_s"] for mode in modes]
        bars = axis.bar(labels, values, color=colors)
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        axis.set_title(title)
        axis.set_ylabel("output tokens/s")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Speculation is workload-dependent")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_final_csv(cases: dict[str, Any], output: Path) -> None:
    fieldnames = [
        "request_rate",
        "system",
        "throughput_tok_s",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "ttft_p95_ms",
        "tpot_p50_ms",
        "tpot_p90_ms",
        "tpot_p95_ms",
        "runs",
        "requests_per_run",
    ]
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for case, rate in (("normal-low", 2), ("normal-loaded", 8)):
            for system in ("light", "vllm_eager"):
                row = cases[case][system]
                writer.writerow(
                    {
                        "request_rate": rate,
                        "system": system,
                        **{key: row[key] for key in fieldnames[2:]},
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results_root = args.repo_root / "benchmarks" / "remote_5090" / "results"
    final = _final_results(results_root)
    speculation = _speculation_results(results_root)
    summary = {
        "final": final,
        "speculation": speculation,
        "provenance": {
            "light_sha": "192d05061e2222116078fb05fb46e73d75b7b375",
            "vllm_mode": "eager",
            "final_runs_per_system_and_rate": 3,
            "percentile_method": (
                "linear interpolation within each 64-request run, then median across three runs"
            ),
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_final_csv(final, args.output / "final_metrics.csv")
    _plot_final_throughput(final, args.output / "01_final_throughput.png")
    _plot_final_latencies(final, args.output / "02_final_latency_percentiles.png")
    _plot_speculation(speculation, args.output / "03_speculation_workload_dependence.png")


if __name__ == "__main__":
    main()
