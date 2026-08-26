from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CASES = ("fixed", "normal-low", "normal-loaded")
METRICS = (
    "output_tokens_per_s",
    "ttft_p50_s",
    "ttft_p95_s",
    "tpot_p50_s",
    "tpot_p95_s",
    "max_itl_p95_s",
    "e2e_p95_s",
)


@dataclass(frozen=True)
class Configuration:
    key: str
    label: str
    directory: Path
    mode: str
    system: str
    tensor_parallel_size: int


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _run(configuration: Configuration, case: str, run: int) -> dict[str, Any]:
    path = configuration.directory / f"{configuration.mode}-{case}-r{run}.json"
    payload = _load(path)
    summary = payload["summary"]
    if int(summary["failed_requests"]) != 0:
        raise ValueError(f"benchmark run contains failed requests: {path}")
    return {
        "run": run,
        "path": str(path),
        "requests": int(summary["requests"]),
        "output_tokens": int(summary["output_tokens"]),
        "_request_shapes": tuple(
            sorted(
                (
                    str(result["request_id"]),
                    int(result["prompt_tokens"]),
                    int(result["requested_output_tokens"]),
                    (
                        int(result["target_output_tokens"])
                        if result["target_output_tokens"] is not None
                        else None
                    ),
                    int(result["output_tokens"]),
                )
                for result in payload["results"]
            )
        ),
        **{metric: float(summary[metric]) for metric in METRICS},
    }


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    request_counts = {run["requests"] for run in runs}
    output_counts = {run["output_tokens"] for run in runs}
    if len(request_counts) != 1 or len(output_counts) != 1:
        raise ValueError("matched runs must contain the same request and output token counts")
    if len({run["_request_shapes"] for run in runs}) != 1:
        raise ValueError("repeated runs must execute the same per-request token counts")
    return {
        "requests": runs[0]["requests"],
        "output_tokens": runs[0]["output_tokens"],
        **{metric: _median([run[metric] for run in runs]) for metric in METRICS},
    }


def _validate_cross_configuration_work(records: dict[str, dict[str, Any]], runs: int) -> None:
    """比较性能前先证明四种配置执行了相同的逐请求工作量。"""

    baseline = records["light_tp1"]
    for case in CASES:
        for run_index in range(runs):
            expected = baseline["runs"][case][run_index]["_request_shapes"]
            for key, record in records.items():
                actual = record["runs"][case][run_index]["_request_shapes"]
                if actual != expected:
                    raise ValueError(
                        f"benchmark work differs from light_tp1: {key}, {case}, run {run_index + 1}"
                    )

    for record in records.values():
        for case_runs in record["runs"].values():
            for run in case_runs:
                del run["_request_shapes"]


def _gpu_summary(configuration: Configuration) -> dict[str, Any]:
    path = configuration.directory / f"{configuration.mode}-runtime-gpu.csv"
    samples: dict[int, list[dict[str, float]]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            index = int(row["index"].strip())
            samples.setdefault(index, []).append(
                {
                    "memory_used_mib": float(row["memory_used_mib"]),
                    "gpu_util_percent": float(row["gpu_util_percent"]),
                    "power_watts": float(row["power_watts"]),
                    "sm_clock_mhz": float(row["sm_clock_mhz"]),
                }
            )
    active = range(configuration.tensor_parallel_size)
    per_gpu = {
        str(index): {
            "peak_memory_used_mib": max(row["memory_used_mib"] for row in samples[index]),
            "gpu_util_p50_percent": _median([row["gpu_util_percent"] for row in samples[index]]),
            "peak_power_watts": max(row["power_watts"] for row in samples[index]),
            "sm_clock_p50_mhz": _median([row["sm_clock_mhz"] for row in samples[index]]),
        }
        for index in active
    }
    return {
        "path": str(path),
        "per_gpu": per_gpu,
        "peak_memory_per_gpu_mib": max(value["peak_memory_used_mib"] for value in per_gpu.values()),
        "sum_of_per_gpu_peaks_mib": sum(
            value["peak_memory_used_mib"] for value in per_gpu.values()
        ),
    }


def _configurations(input_root: Path) -> tuple[Configuration, ...]:
    return (
        Configuration(
            "light_tp1", "light-vllm TP=1", input_root / "light-tp1", "light-strict", "light", 1
        ),
        Configuration(
            "vllm_tp1",
            "vLLM eager TP=1",
            input_root / "vllm-eager-tp1",
            "vllm-eager",
            "vllm",
            1,
        ),
        Configuration(
            "light_tp2", "light-vllm TP=2", input_root / "light-tp2", "light-strict", "light", 2
        ),
        Configuration(
            "vllm_tp2",
            "vLLM eager TP=2",
            input_root / "vllm-eager-tp2",
            "vllm-eager",
            "vllm",
            2,
        ),
    )


def _comparisons(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"light_vs_vllm": {}, "tp_scaling": {}}
    for case in CASES:
        result["light_vs_vllm"][case] = {}
        for tp in (1, 2):
            light = records[f"light_tp{tp}"]["cases"][case]
            vllm = records[f"vllm_tp{tp}"]["cases"][case]
            result["light_vs_vllm"][case][f"tp{tp}"] = {
                "throughput_ratio": (light["output_tokens_per_s"] / vllm["output_tokens_per_s"]),
                "ttft_p95_delta_ms": (light["ttft_p95_s"] - vllm["ttft_p95_s"]) * 1000,
                "tpot_p50_ratio": light["tpot_p50_s"] / vllm["tpot_p50_s"],
            }
        result["tp_scaling"][case] = {}
        for system in ("light", "vllm"):
            tp1 = records[f"{system}_tp1"]["cases"][case]
            tp2 = records[f"{system}_tp2"]["cases"][case]
            result["tp_scaling"][case][system] = {
                "throughput_speedup": (tp2["output_tokens_per_s"] / tp1["output_tokens_per_s"]),
                "tpot_speedup": tp1["tpot_p50_s"] / tp2["tpot_p50_s"],
            }
    return result


def _write_csv(summary: dict[str, Any], path: Path) -> None:
    fields = (
        "configuration",
        "system",
        "tensor_parallel_size",
        "case",
        "requests",
        "output_tokens",
        "output_tokens_per_s",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "tpot_p50_ms",
        "tpot_p95_ms",
        "max_itl_p95_ms",
        "e2e_p95_ms",
        "peak_memory_per_gpu_mib",
        "sum_of_per_gpu_peaks_mib",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in summary["configurations"].values():
            for case in CASES:
                median = record["cases"][case]
                writer.writerow(
                    {
                        "configuration": record["label"],
                        "system": record["system"],
                        "tensor_parallel_size": record["tensor_parallel_size"],
                        "case": case,
                        "requests": median["requests"],
                        "output_tokens": median["output_tokens"],
                        "output_tokens_per_s": median["output_tokens_per_s"],
                        "ttft_p50_ms": median["ttft_p50_s"] * 1000,
                        "ttft_p95_ms": median["ttft_p95_s"] * 1000,
                        "tpot_p50_ms": median["tpot_p50_s"] * 1000,
                        "tpot_p95_ms": median["tpot_p95_s"] * 1000,
                        "max_itl_p95_ms": median["max_itl_p95_s"] * 1000,
                        "e2e_p95_ms": median["e2e_p95_s"] * 1000,
                        "peak_memory_per_gpu_mib": record["gpu"]["peak_memory_per_gpu_mib"],
                        "sum_of_per_gpu_peaks_mib": record["gpu"]["sum_of_per_gpu_peaks_mib"],
                    }
                )


def _bar_chart(
    axis,
    records: list[dict[str, Any]],
    case: str,
    metric: str,
    title: str,
    *,
    scale: float = 1.0,
) -> None:
    colors = ("#2F6B9A", "#7A7A7A", "#74A9CF", "#B9B9B9")
    values = [record["cases"][case][metric] * scale for record in records]
    bars = axis.bar(range(len(records)), values, color=colors)
    axis.set_title(title)
    axis.set_xticks(range(len(records)), [record["short_label"] for record in records])
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)


def _plot_summary(summary: dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    records = list(summary["configurations"].values())
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for axis, case, title in zip(
        axes,
        CASES,
        ("Fixed decode", "Production 2 req/s", "Production 8 req/s"),
        strict=True,
    ):
        _bar_chart(axis, records, case, "output_tokens_per_s", title)
        axis.set_ylabel("output tokens/s")
    figure.suptitle("Qwen2.5-Coder-7B BF16 throughput on 2x RTX 5090 (no CUDA P2P)")
    figure.savefig(output_dir / "01_tp_throughput.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for row, (case, load) in enumerate((("normal-low", "2 req/s"), ("normal-loaded", "8 req/s"))):
        _bar_chart(axes[row, 0], records, case, "ttft_p95_s", f"{load} TTFT P95", scale=1000)
        axes[row, 0].set_ylabel("ms/request")
        _bar_chart(axes[row, 1], records, case, "tpot_p50_s", f"{load} TPOT P50", scale=1000)
        axes[row, 1].set_ylabel("ms/token")
    figure.suptitle("Latency medians across repeated runs")
    figure.savefig(output_dir / "02_tp_latency.png", dpi=180)
    plt.close(figure)

    comparisons = summary["comparisons"]["tp_scaling"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    labels = ("Fixed", "2 req/s", "8 req/s")
    x = range(len(CASES))
    width = 0.34
    for offset, system, label, color in (
        (-width / 2, "light", "light-vllm", "#2F6B9A"),
        (width / 2, "vllm", "vLLM eager", "#7A7A7A"),
    ):
        throughput = [comparisons[case][system]["throughput_speedup"] for case in CASES]
        tpot = [comparisons[case][system]["tpot_speedup"] for case in CASES]
        axes[0].bar([value + offset for value in x], throughput, width, label=label, color=color)
        axes[1].bar([value + offset for value in x], tpot, width, label=label, color=color)
    for axis, title in zip(axes, ("TP=2 / TP=1 throughput", "TP=1 / TP=2 TPOT"), strict=True):
        axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
        axis.set_title(title)
        axis.set_xticks(list(x), labels)
        axis.set_ylabel("speedup")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.suptitle("Tensor-parallel scaling efficiency")
    figure.savefig(output_dir / "03_tp_scaling.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    memory = [record["gpu"]["peak_memory_per_gpu_mib"] / 1024 for record in records]
    bars = axis.bar(
        range(len(records)),
        memory,
        color=("#2F6B9A", "#7A7A7A", "#74A9CF", "#B9B9B9"),
    )
    axis.set_xticks(range(len(records)), [record["short_label"] for record in records])
    axis.set_ylabel("peak GiB per active GPU")
    axis.set_title("Peak device memory during the full benchmark")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.2f", padding=3)
    figure.savefig(output_dir / "04_tp_memory.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    records: dict[str, dict[str, Any]] = {}
    for configuration in _configurations(args.input_root):
        runs_by_case = {
            case: [_run(configuration, case, run) for run in range(1, args.runs + 1)]
            for case in CASES
        }
        environment_path = (
            configuration.directory / f"{configuration.mode}-runtime-environment.json"
        )
        environment = _load(environment_path)
        if int(environment["tensor_parallel_size"]) != configuration.tensor_parallel_size:
            raise ValueError(f"environment TP size does not match: {environment_path}")
        records[configuration.key] = {
            "label": configuration.label,
            "short_label": f"{configuration.system} TP{configuration.tensor_parallel_size}",
            "system": configuration.system,
            "tensor_parallel_size": configuration.tensor_parallel_size,
            "runs": runs_by_case,
            "cases": {case: _summarize_runs(runs) for case, runs in runs_by_case.items()},
            "gpu": _gpu_summary(configuration),
            "environment": {"path": str(environment_path), **environment},
        }

    _validate_cross_configuration_work(records, args.runs)
    summary = {
        "method": {
            "runs": args.runs,
            "aggregation": "median of per-run metrics",
            "cases": list(CASES),
            "platform_note": "2x RTX 5090, NODE topology, CUDA P2P unavailable",
        },
        "configurations": records,
        "comparisons": _comparisons(records),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_csv(summary, args.output_dir / "metrics.csv")
    _plot_summary(summary, args.output_dir)


if __name__ == "__main__":
    main()
