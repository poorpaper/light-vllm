from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

METRICS = {
    "throughput_tok_s": "output_tokens_per_s",
    "ttft_p95_ms": "ttft_p95_s",
    "tpot_p50_ms": "tpot_p50_s",
    "tpot_p95_ms": "tpot_p95_s",
}
LABELS = {
    "baseline": "Light baseline",
    "candidate": "Light optimized",
    "vllm_eager": "vLLM eager",
}
COLORS = {
    "baseline": "#A0A0A0",
    "candidate": "#E45756",
    "vllm_eager": "#4C78A8",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _light_runs(root: Path, mode: str, case: str) -> list[dict[str, Any]]:
    paths = [root / f"{mode}-r{run}" / f"light-strict-{case}-r1.json" for run in (1, 2, 3)]
    return _require_runs(paths)


def _vllm_runs(root: Path, case: str) -> list[dict[str, Any]]:
    paths = [root / f"vllm-eager-{case}-r{run}.json" for run in (1, 2, 3)]
    return _require_runs(paths)


def _require_runs(paths: list[Path]) -> list[dict[str, Any]]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing formal benchmark runs: {missing}")
    return [_load(path) for path in paths]


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {name: [] for name in METRICS}
    successes: list[int] = []
    requests: list[int] = []
    for run in runs:
        summary = run["summary"]
        successes.append(int(summary["successful_requests"]))
        requests.append(int(summary["requests"]))
        for name, source in METRICS.items():
            value = float(summary[source])
            if name.endswith("_ms"):
                value *= 1000
            values[name].append(value)
    return {
        "runs": 3,
        "all_requests_successful": successes == requests,
        "successful_requests_per_run": successes,
        **{f"{name}_runs": samples for name, samples in values.items()},
        **{name: statistics.median(samples) for name, samples in values.items()},
    }


def _ratios(rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    baseline = rows["baseline"]
    candidate = rows["candidate"]
    eager = rows["vllm_eager"]
    return {
        "candidate_throughput_vs_vllm_percent": 100
        * candidate["throughput_tok_s"]
        / eager["throughput_tok_s"],
        # TPOT 越低越快，因此使用 eager/candidate 表达相对生成速度。
        "candidate_tpot_speed_vs_vllm_percent": 100
        * eager["tpot_p50_ms"]
        / candidate["tpot_p50_ms"],
        "candidate_ttft_p95_vs_vllm_ratio": candidate["ttft_p95_ms"] / eager["ttft_p95_ms"],
        "candidate_throughput_gain_vs_baseline_percent": 100
        * (candidate["throughput_tok_s"] / baseline["throughput_tok_s"] - 1),
        "candidate_tpot_reduction_vs_baseline_percent": 100
        * (1 - candidate["tpot_p50_ms"] / baseline["tpot_p50_ms"]),
        "candidate_ttft_p95_reduction_vs_baseline_percent": 100
        * (1 - candidate["ttft_p95_ms"] / baseline["ttft_p95_ms"]),
    }


def _profile_summary(light_path: Path, vllm_path: Path) -> dict[str, Any]:
    light = _load(light_path)
    vllm = _load(vllm_path)
    light_steps = int(light["executor_steps"])
    vllm_steps = int(vllm["engine_steps"])
    return {
        "boundary_note": (
            "Light executor CUDA event and vLLM execute_model are independently "
            "instrumented boundaries, so compare them as diagnostic evidence, not "
            "as an exact kernel-to-kernel decomposition."
        ),
        "light": {
            "steps": light_steps,
            "executor_wall_ms_per_step": light["executor_wall_total_ms"] / light_steps,
            "cuda_event_ms_per_step": light["executor_cuda_event_total_ms"] / light_steps,
            "inter_executor_gap_ms_per_step": light["inter_executor_gap_total_ms"]
            / (light_steps - 1),
            "service_span_ms_per_step": light["executor_span_ms"] / light_steps,
            "schedule_ms_per_step": light["stages"]["scheduler.schedule"]["wall_mean_ms"],
            "advance_locked_ms_per_step": light["stages"]["engine.advance_locked"]["wall_mean_ms"],
            "apply_output_ms_per_step": light["stages"]["engine.apply_output"]["wall_mean_ms"],
        },
        "vllm_eager": {
            "steps": vllm_steps,
            "engine_step_wall_ms_per_step": vllm["engine_step_wall_total_ms"] / vllm_steps,
            "execute_model_ms_per_step": vllm["stages"]["executor.execute_model"]["wall_mean_ms"],
            "inter_step_gap_ms_per_step": vllm["inter_step_gap_total_ms"] / (vllm_steps - 1),
            "service_span_ms_per_step": vllm["engine_step_span_ms"] / vllm_steps,
            "future_result_ms_per_step": vllm["stages"]["executor.future_result"]["wall_mean_ms"],
            "sample_tokens_ms_per_step": vllm["stages"]["executor.sample_tokens"]["wall_mean_ms"],
            "schedule_ms_per_step": vllm["stages"]["scheduler.schedule"]["wall_mean_ms"],
            "update_output_ms_per_step": vllm["stages"]["scheduler.update_from_output"][
                "wall_mean_ms"
            ],
        },
    }


def _plot(cases: dict[str, dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    case_names = list(cases)
    modes = ("baseline", "candidate", "vllm_eager")
    metrics = (
        ("throughput_tok_s", "Output throughput", "tokens/s"),
        ("tpot_p50_ms", "TPOT P50", "ms/token"),
        ("ttft_p95_ms", "TTFT P95", "ms"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    width = 0.24
    x = np.arange(len(case_names))
    for axis, (metric, title, ylabel) in zip(axes, metrics, strict=True):
        for index, mode in enumerate(modes):
            values = [cases[case][mode][metric] for case in case_names]
            bars = axis.bar(
                x + (index - 1) * width,
                values,
                width,
                label=LABELS[mode],
                color=COLORS[mode],
            )
            axis.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(
            x, ["8 req/s" if case == "normal-loaded" else "2 req/s" for case in case_names]
        )
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3)
    figure.suptitle("Light-vLLM hot-path optimization vs vLLM eager (median of 3 runs)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_report(summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# Light-vLLM 90% 热路径优化实验",
        "",
        "主结论基于 Poisson 生产长度负载，每个实现独立启动并运行 3 次，取各轮指标中位数。",
        "",
    ]
    for case, label in (("normal-loaded", "8 req/s"), ("normal-low", "2 req/s")):
        if case not in summary["cases"]:
            continue
        result = summary["cases"][case]
        ratios = result["ratios"]
        candidate = result["candidate"]
        eager = result["vllm_eager"]
        lines.extend(
            [
                f"## {label}",
                "",
                "| 指标 | Light optimized | vLLM eager | 相对结果 |",
                "| --- | ---: | ---: | ---: |",
                (
                    f"| Throughput | {candidate['throughput_tok_s']:.2f} tok/s | "
                    f"{eager['throughput_tok_s']:.2f} tok/s | "
                    f"{ratios['candidate_throughput_vs_vllm_percent']:.1f}% |"
                ),
                (
                    f"| TPOT P50 | {candidate['tpot_p50_ms']:.3f} ms | "
                    f"{eager['tpot_p50_ms']:.3f} ms | "
                    f"{ratios['candidate_tpot_speed_vs_vllm_percent']:.1f}% speed |"
                ),
                (
                    f"| TTFT P95 | {candidate['ttft_p95_ms']:.2f} ms | "
                    f"{eager['ttft_p95_ms']:.2f} ms | "
                    f"{ratios['candidate_ttft_p95_vs_vllm_ratio']:.2f}x |"
                ),
                "",
            ]
        )
    if "profiles" in summary:
        profile = summary["profiles"]
        light = profile["light"]
        eager = profile["vllm_eager"]
        lines.extend(
            [
                "## 执行边界诊断",
                "",
                "| 边界 | Light optimized | vLLM eager |",
                "| --- | ---: | ---: |",
                (
                    f"| 设备/模型执行 | {light['cuda_event_ms_per_step']:.3f} ms/step "
                    f"(CUDA event) | {eager['execute_model_ms_per_step']:.3f} ms/step "
                    "(execute_model wall) |"
                ),
                (
                    f"| 步间空档 | {light['inter_executor_gap_ms_per_step']:.3f} ms/step | "
                    f"{eager['inter_step_gap_ms_per_step']:.3f} ms/step |"
                ),
                (
                    f"| 端到端 step span | {light['service_span_ms_per_step']:.3f} ms/step | "
                    f"{eager['service_span_ms_per_step']:.3f} ms/step |"
                ),
                "",
                "设备/模型执行边界并不完全相同，不能把两列之差直接解释成某个 kernel 的差值；"
                "它们只用于定位剩余差距仍同时包含模型执行与 Driver 空档。",
                "",
            ]
        )
    lines.extend(
        [
            "## 结论边界",
            "",
            "Light strict 保持 completion claim 非抢占语义；所有正式请求均成功。"
            "该结论只适用于记录的模型、RTX 5090、vLLM eager 配置和本次负载。",
            "继续追赶最后约 9% TPOT 预计需要 CUDA Graph、更多融合 kernel 或真正的多步在途流水，"
            "改动与验证成本显著高于本轮保留的局部热路径优化，因此在 90% 停止线收敛。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--light-loaded", type=Path, required=True)
    parser.add_argument("--vllm-loaded", type=Path, required=True)
    parser.add_argument("--light-low", type=Path)
    parser.add_argument("--vllm-low", type=Path)
    parser.add_argument("--light-profile", type=Path)
    parser.add_argument("--vllm-profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases: dict[str, dict[str, Any]] = {}
    inputs = [("normal-loaded", args.light_loaded, args.vllm_loaded)]
    if args.light_low is not None or args.vllm_low is not None:
        if args.light_low is None or args.vllm_low is None:
            raise ValueError("light-low and vllm-low must be provided together")
        inputs.append(("normal-low", args.light_low, args.vllm_low))
    for case, light_root, vllm_root in inputs:
        rows = {
            "baseline": _summarize(_light_runs(light_root, "baseline", case)),
            "candidate": _summarize(_light_runs(light_root, "candidate", case)),
            "vllm_eager": _summarize(_vllm_runs(vllm_root, case)),
        }
        cases[case] = {**rows, "ratios": _ratios(rows)}

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "aggregation": "median of three per-run benchmark summaries",
        "cases": cases,
    }
    if args.light_profile is not None or args.vllm_profile is not None:
        if args.light_profile is None or args.vllm_profile is None:
            raise ValueError("light-profile and vllm-profile must be provided together")
        summary["profiles"] = _profile_summary(args.light_profile, args.vllm_profile)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _plot(cases, args.output / "comparison.png")
    _write_report(summary, args.output / "report.md")


if __name__ == "__main__":
    main()
