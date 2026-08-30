from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Any

METRICS = {
    "output_tokens_per_s": ("Output throughput", "tok/s", True),
    "ttft_p50_s": ("TTFT P50", "ms", False),
    "tpot_p50_s": ("TPOT P50", "ms/token", False),
}
BACKENDS = ("light-vllm", "vllm")
EXPECTED_RUNS = 6
EXPECTED_SHAPES = {
    "baseline-256-64": (64, 256, 64),
    "decode-steady-16-512": (16, 16, 512),
}
EXPECTED_WORKLOADS = set(EXPECTED_SHAPES)


def _load_runs(result_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    paths = sorted((result_dir / "raw").glob("*/*-r[12].json"))
    if not paths:
        raise ValueError(f"no measurement JSON found under {result_dir / 'raw'}")

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        backend = str(payload["metadata"]["backend"])
        workload = str(payload["metadata"]["workload"])
        if backend not in BACKENDS:
            raise ValueError(f"unexpected backend in {path}: {backend}")
        if workload not in EXPECTED_SHAPES:
            raise ValueError(f"unexpected workload in {path}: {workload}")
        summary = payload["summary"]
        if summary["failed_requests"] or summary["successful_requests"] != summary["requests"]:
            raise ValueError(f"benchmark run is incomplete: {path}")
        if summary["output_limit_hits"] != summary["requests"]:
            raise ValueError(f"benchmark run ended before the requested output length: {path}")
        expected_requests, prompt_tokens, output_tokens = EXPECTED_SHAPES[workload]
        results = payload["results"]
        request_ids = [str(result["request_id"]) for result in results]
        if len(results) != expected_requests or len(set(request_ids)) != expected_requests:
            raise ValueError(f"benchmark request count or IDs do not match {workload}: {path}")
        if any(
            result["prompt_tokens"] != prompt_tokens
            or result["requested_output_tokens"] != output_tokens
            for result in results
        ):
            raise ValueError(f"benchmark request shape does not match {workload}: {path}")
        grouped[(workload, backend)].append({"path": path, "payload": payload})

    workloads = {workload for workload, _ in grouped}
    if workloads != EXPECTED_WORKLOADS:
        raise ValueError(
            f"expected workloads {sorted(EXPECTED_WORKLOADS)}, found {sorted(workloads)}"
        )
    for workload in sorted(workloads):
        for backend in BACKENDS:
            count = len(grouped[(workload, backend)])
            if count != EXPECTED_RUNS:
                raise ValueError(
                    f"expected {EXPECTED_RUNS} runs for {workload}/{backend}, found {count}"
                )
    return grouped


def _token_outputs(payload: dict[str, Any]) -> dict[str, list[int]]:
    return {
        str(result["request_id"]): [int(token) for token in result["generated_token_ids"]]
        for result in payload["results"]
    }


def _compare_output_pair(
    left: dict[str, list[int]],
    right: dict[str, list[int]],
    workload: str,
) -> dict[str, Any]:
    if left.keys() != right.keys():
        raise ValueError(f"request IDs differ between output runs for {workload}")

    exact_requests = 0
    matching_positions = 0
    total_positions = 0
    for request_id in left:
        left_tokens = left[request_id]
        right_tokens = right[request_id]
        if len(left_tokens) != len(right_tokens):
            raise ValueError(f"output lengths differ for {workload}/{request_id}")
        exact_requests += left_tokens == right_tokens
        matching_positions += sum(
            left_token == right_token
            for left_token, right_token in zip(left_tokens, right_tokens, strict=True)
        )
        total_positions += len(left_tokens)
    return {
        "exact_output": exact_requests == len(left),
        "exact_request_percent": exact_requests / len(left) * 100,
        "matching_token_percent": matching_positions / total_positions * 100,
    }


def _pairwise_output_summary(
    left_runs: list[dict[str, Any]],
    right_runs: list[dict[str, Any]],
    workload: str,
    *,
    same_backend: bool,
) -> dict[str, Any]:
    """汇总所有轮次对，避免挑选一轮掩盖 batch-sensitive greedy 差异。"""

    left_outputs = [_token_outputs(run["payload"]) for run in left_runs]
    right_outputs = [_token_outputs(run["payload"]) for run in right_runs]
    if same_backend:
        pairs = combinations(left_outputs, 2)
        distinct_maps = len(
            {
                hashlib.sha256(
                    json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                for output in left_outputs
            }
        )
    else:
        pairs = product(left_outputs, right_outputs)
        distinct_maps = None
    comparisons = [_compare_output_pair(left, right, workload) for left, right in pairs]
    request_percent = [comparison["exact_request_percent"] for comparison in comparisons]
    token_percent = [comparison["matching_token_percent"] for comparison in comparisons]
    result = {
        "run_pairs": len(comparisons),
        "exact_output_pairs": sum(comparison["exact_output"] for comparison in comparisons),
        "exact_request_percent": {
            "median": statistics.median(request_percent),
            "min": min(request_percent),
            "max": max(request_percent),
        },
        "matching_token_percent": {
            "median": statistics.median(token_percent),
            "min": min(token_percent),
            "max": max(token_percent),
        },
    }
    if distinct_maps is not None:
        result["distinct_output_maps"] = distinct_maps
    return result


def _metric_value(summary: dict[str, Any], metric: str) -> float:
    value = summary[metric]
    if value is None:
        raise ValueError(f"metric {metric} is missing")
    number = float(value)
    return number * 1000 if metric in {"ttft_p50_s", "tpot_p50_s"} else number


def _summarize(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    workloads: dict[str, dict[str, Any]] = {}
    for workload, backend in sorted(grouped):
        runs = grouped[(workload, backend)]
        backend_summary: dict[str, Any] = {
            "runs": [str(run["path"].name) for run in runs],
            "metrics": {},
        }
        for metric in METRICS:
            values = [_metric_value(run["payload"]["summary"], metric) for run in runs]
            backend_summary["metrics"][metric] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "values": values,
            }
        workloads.setdefault(workload, {})[backend] = backend_summary

    for systems in workloads.values():
        light = systems["light-vllm"]["metrics"]
        vllm = systems["vllm"]["metrics"]
        systems["relative_percent"] = {
            metric: (light[metric]["median"] / vllm[metric]["median"] - 1) * 100
            for metric in METRICS
        }
    return {
        "measurement_runs_per_backend": EXPECTED_RUNS,
        "workloads": workloads,
        "output_stability": {
            workload: {
                backend: _pairwise_output_summary(
                    grouped[(workload, backend)],
                    grouped[(workload, backend)],
                    workload,
                    same_backend=True,
                )
                for backend in BACKENDS
            }
            for workload in sorted(workloads)
        },
        "output_comparison": {
            workload: _pairwise_output_summary(
                grouped[(workload, "light-vllm")],
                grouped[(workload, "vllm")],
                workload,
                same_backend=False,
            )
            for workload in sorted(workloads)
        },
    }


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("workload", "backend", "metric", "median", "min", "max", "unit"))
        for workload, systems in summary["workloads"].items():
            for backend in BACKENDS:
                for metric, (_, unit, _) in METRICS.items():
                    values = systems[backend]["metrics"][metric]
                    writer.writerow(
                        (
                            workload,
                            backend,
                            metric,
                            values["median"],
                            values["min"],
                            values["max"],
                            unit,
                        )
                    )


def _plot(path: Path, summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    workloads = list(summary["workloads"])
    labels = [name.replace("baseline-", "").replace("decode-steady-", "") for name in workloads]
    x = np.arange(len(workloads), dtype=float)
    width = 0.34
    colors = {"light-vllm": "#2563eb", "vllm": "#f59e0b"}
    figure, axes = plt.subplots(1, len(METRICS), figsize=(13.5, 4.2), constrained_layout=True)

    for axis, (metric, (title, unit, higher_is_better)) in zip(axes, METRICS.items(), strict=True):
        for offset, backend in ((-width / 2, "light-vllm"), (width / 2, "vllm")):
            medians = []
            lower = []
            upper = []
            for workload in workloads:
                values = summary["workloads"][workload][backend]["metrics"][metric]
                medians.append(values["median"])
                lower.append(values["median"] - values["min"])
                upper.append(values["max"] - values["median"])
            bars = axis.bar(
                x + offset,
                medians,
                width,
                yerr=[lower, upper],
                capsize=3,
                label=backend,
                color=colors[backend],
                alpha=0.9,
            )
            axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        direction = "higher is better" if higher_is_better else "lower is better"
        axis.set_title(f"{title}\n({direction})")
        axis.set_ylabel(unit)
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("AWQ W4A16 end-to-end comparison (median and six-run range)")
    figure.savefig(path.with_suffix(".png"), dpi=180)
    figure.savefig(path.with_suffix(".svg"))
    plt.close(figure)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# AWQ W4A16 与 vLLM 端到端对照",
        "",
        "这是两组固定 burst 压测：64 个 256→64 请求和 16 个 16→512 请求，不代表生产流量。",
        "表中是每个系统 6 轮实测的中位数，括号内为最小值到最大值；两边使用 greedy、FP16、",
        "同一 AWQ checkpoint。完整环境和执行顺序见 `environment.json` 与 `commands.txt`。",
        "",
        "| workload | 系统 | 输出吞吐 tok/s | TTFT P50 ms | TPOT P50 ms/token |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for workload, systems in summary["workloads"].items():
        for backend in BACKENDS:
            values = systems[backend]["metrics"]
            cells = []
            for metric in METRICS:
                item = values[metric]
                cells.append(f"{item['median']:.2f} ({item['min']:.2f}–{item['max']:.2f})")
            lines.append(f"| {workload} | {backend} | " + " | ".join(cells) + " |")
    lines.extend(("", "同一实现的 greedy 输出稳定性（全部 15 个轮次对）："))
    for workload, systems in summary["output_stability"].items():
        for backend in BACKENDS:
            stability = systems[backend]
            token_match = stability["matching_token_percent"]
            lines.append(
                f"- {workload} / {backend}: {stability['distinct_output_maps']} 个输出 map；"
                f"{stability['exact_output_pairs']}/{stability['run_pairs']} 个轮次对完全一致；"
                f"同位置 token-ID 匹配率中位数 {token_match['median']:.2f}% "
                f"（{token_match['min']:.2f}%–{token_match['max']:.2f}%）。"
            )
    lines.extend(("", "实现间 greedy 输出对照（全部 36 个跨实现轮次对）："))
    for workload, comparison in summary["output_comparison"].items():
        request_match = comparison["exact_request_percent"]
        token_match = comparison["matching_token_percent"]
        lines.append(
            f"- {workload}: 完全一致请求比例中位数 {request_match['median']:.2f}% "
            f"（{request_match['min']:.2f}%–{request_match['max']:.2f}%）；"
            f"同位置 token-ID 匹配率中位数 {token_match['median']:.2f}% "
            f"（{token_match['min']:.2f}%–{token_match['max']:.2f}%）。"
        )
    lines.extend(
        (
            "",
            "所有测量轮次必须零请求失败、达到固定输出长度并保持相同请求集合；否则分析脚本"
            "直接失败。轮次内或实现间的 token 差异不会被丢弃，而是用全部轮次对如实统计。",
            "同位置 token-ID 匹配率不是质量指标；logits、PPL 或任务质量需要由独立质量验收给出，"
            "不能由这组性能压测推断。",
            "该结果只覆盖记录中的机器、模型和 workload，不能外推为所有场景下普遍优于 vLLM。",
            "",
            "![AWQ W4A16 与 vLLM 对照](figures/awq-vllm.png)",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_hashes(result_dir: Path) -> None:
    excluded = {"SHA256SUMS"}
    paths = sorted(
        path for path in result_dir.rglob("*") if path.is_file() and path.name not in excluded
    )
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(result_dir).as_posix()}")
    (result_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    grouped = _load_runs(result_dir)
    summary = _summarize(grouped)
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(result_dir / "summary.csv", summary)
    figures = result_dir / "figures"
    figures.mkdir(exist_ok=True)
    _plot(figures / "awq-vllm", summary)
    _write_report(result_dir / "REPORT.md", summary)
    _write_hashes(result_dir)


if __name__ == "__main__":
    main()
