from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_driver_sampler_staging import (
    MODES,
    RATES,
    _formal_runs,
    _plot_throughput,
    _plot_ttft,
    _summarize_mode,
    _write_csv,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_matrix(root: Path) -> list[dict[str, Any]]:
    rows = [
        _summarize_mode(rate, mode, _formal_runs(root, rate, mode))
        for rate in RATES
        for mode in MODES
    ]
    eager = {
        int(row["rate_req_s"]): float(row["throughput_tok_s_median"])
        for row in rows
        if row["mode"] == "vllm-eager"
    }
    for row in rows:
        row["throughput_vs_vllm_eager"] = (
            float(row["throughput_tok_s_median"]) / eager[int(row["rate_req_s"])]
        )
    return rows


def _read_ab(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in ("baseline", "candidate"):
        runs = [
            _load(root / f"{label}-r{run}" / "light-strict-normal-loaded-r1.json")
            for run in (1, 2, 3)
        ]
        summary = _summarize_mode(8, "light-strict", runs)
        result[label] = summary
    result["throughput_change_percent"] = 100 * (
        float(result["candidate"]["throughput_tok_s_median"])
        / float(result["baseline"]["throughput_tok_s_median"])
        - 1
    )
    result["ttft_p95_change_percent"] = 100 * (
        float(result["candidate"]["ttft_ms"]["p95"]) / float(result["baseline"]["ttft_ms"]["p95"])
        - 1
    )
    return result


def _read_profile(path: Path) -> dict[str, Any]:
    payload = _load(path)
    steps = int(payload["executor_steps"])
    gap = payload["inter_executor_gap_breakdown"]
    gap_steps = int(gap["steps"])
    hot_edges = []
    for step in payload["steps"]:
        edge = {
            "event_loop_resume": step.get("worker_finish_to_event_loop_resume_ms"),
            "control_path": step.get("event_loop_resume_to_next_submit_ms"),
            "dispatch": step.get("next_submit_to_worker_start_ms"),
        }
        if all(value is not None for value in edge.values()) and sum(edge.values()) < 5.0:
            hot_edges.append(edge)
    result = {
        "path": str(path),
        "steps": steps,
        "service_ms_per_step": float(payload["executor_span_ms"]) / steps,
        "executor_ms_per_step": float(payload["executor_wall_total_ms"]) / steps,
        "cuda_event_ms_per_step": float(payload["executor_cuda_event_total_ms"]) / steps,
        "gap_ms_per_edge": float(payload["inter_executor_gap_total_ms"]) / gap_steps,
        "event_loop_resume_ms_per_edge": float(gap["event_loop_resume_total_ms"]) / gap_steps,
        "control_path_ms_per_edge": float(gap["control_path_total_ms"]) / gap_steps,
        "dispatch_ms_per_edge": float(gap["next_dispatch_total_ms"]) / gap_steps,
        "hot_edge_definition": "consecutive executor gap below 5 ms",
        "hot_edges": len(hot_edges),
        **{
            f"hot_{name}_ms_per_edge": statistics.fmean(float(edge[name]) for edge in hot_edges)
            for name in ("event_loop_resume", "control_path", "dispatch")
        },
    }
    control = payload.get("driver_control_breakdown")
    if control:
        control_steps = int(control["steps"])
        result["driver_control"] = {
            key.removesuffix("_total_ms") + "_ms_per_edge": float(value) / control_steps
            for key, value in control.items()
            if key.endswith("_total_ms")
        }
        hot_control = [
            step["driver_control_ms"]
            for step in payload["steps"]
            if step.get("driver_control_ms") is not None
            and sum(step["driver_control_ms"].values()) < 5.0
        ]
        result["hot_driver_control"] = {
            "steps": len(hot_control),
            **{
                f"{name}_mean_ms_per_edge": statistics.fmean(
                    float(parts[name]) for parts in hot_control
                )
                for name in hot_control[0]
            },
            **{
                f"{name}_p50_ms_per_edge": statistics.median(
                    float(parts[name]) for parts in hot_control
                )
                for name in hot_control[0]
            },
        }
    return result


def _plot_ab(experiments: dict[str, dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(experiments)
    x = np.arange(len(names))
    width = 0.34
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for index, revision in enumerate(("baseline", "candidate")):
        throughput = [experiments[name][revision]["throughput_tok_s_median"] for name in names]
        ttft = [experiments[name][revision]["ttft_ms"]["p95"] for name in names]
        offset = (index - 0.5) * width
        throughput_bars = axes[0].bar(x + offset, throughput, width, label=revision)
        ttft_bars = axes[1].bar(x + offset, ttft, width, label=revision)
        axes[0].bar_label(throughput_bars, fmt="%.1f", padding=3, fontsize=8)
        axes[1].bar_label(ttft_bars, fmt="%.1f", padding=3, fontsize=8)
    axes[0].set_title("8 req/s output throughput")
    axes[0].set_ylabel("Output tokens/s, median of 3 runs")
    axes[1].set_title("8 req/s TTFT P95")
    axes[1].set_ylabel("Milliseconds, 192 pooled requests")
    for axis in axes:
        axis.set_xticks(x, names)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    figure.suptitle("Isolated Driver changes; each pair has its own interleaved A/B")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_gap(profiles: dict[str, dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(profiles)
    x = np.arange(len(names))
    components = (
        ("hot_event_loop_resume_ms_per_edge", "Worker finish -> loop resume"),
        ("hot_control_path_ms_per_edge", "Driver control path"),
        ("hot_dispatch_ms_per_edge", "Submit -> worker start"),
    )
    bottom = np.zeros(len(names))
    figure, axis = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    for key, label in components:
        values = np.array([profiles[name][key] for name in names])
        axis.bar(x, values, bottom=bottom, label=label)
        bottom += values
    axis.set_xticks(x, names)
    axis.set_ylabel("Mean milliseconds per consecutive-step edge")
    axis.set_title("Light Driver hot execution-boundary gap (<5 ms consecutive edges)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    for index, total in enumerate(bottom):
        axis.text(index, total + 0.02, f"{total:.3f}", ha="center", fontsize=9)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _row(rows: list[dict[str, Any]], rate: int, mode: str) -> dict[str, Any]:
    return next(row for row in rows if row["rate_req_s"] == rate and row["mode"] == mode)


def _write_report(summary: dict[str, Any], output: Path) -> None:
    rows = summary["matrix"]
    light = _row(rows, 8, "light-strict")
    eager = _row(rows, 8, "vllm-eager")
    state = summary["isolated_ab"]["coalesced_state_machine"]
    lane = summary["isolated_ab"]["private_execution_lane"]
    profiles = summary["profiles"]
    provenance = summary["provenance"]
    final_profile = profiles["private lane"]
    control = final_profile.get("hot_driver_control", {})
    original_dispatch = profiles["coalesced state"]["hot_dispatch_ms_per_edge"]
    final_dispatch = final_profile["hot_dispatch_ms_per_edge"]
    light_tps = light["throughput_tok_s_median"]
    light_ratio = 100 * light["throughput_vs_vllm_eager"]
    light_ttft = light["ttft_ms"]["p95"]
    eager_ttft = eager["ttft_ms"]["p95"]
    state_tps_change = state["throughput_change_percent"]
    state_ttft_change = state["ttft_p95_change_percent"]
    lane_tps_change = lane["throughput_change_percent"]
    lane_ttft_change = lane["ttft_p95_change_percent"]
    output_processing = control.get("output_processing_mean_ms_per_edge", float("nan"))
    transition = control.get("transition_to_advance_p50_ms_per_edge", float("nan"))
    advance = control.get("advance_locked_mean_ms_per_edge", float("nan"))
    next_entry = control.get("next_execute_entry_mean_ms_per_edge", float("nan"))
    output.write_text(
        f"""# Driver step pipeline：根因、改动与 5090 实验

## 一句话结论

Light 的通用线程池派发确实是可消除损失，但不是全部差距。单在途常驻执行 lane 把下一轮派发从
`{original_dispatch:.3f}` 降到 `{final_dispatch:.3f} ms/step`；最终正常 8 req/s 下 Light strict
达到 `{light_tps:.2f} tok/s`，为 vLLM eager 的 `{light_ratio:.1f}%`，TTFT P95 为
`{light_ttft:.2f} ms`（eager `{eager_ttft:.2f} ms`）。

## 两个隔离改动

| 改动 | 吞吐变化 | TTFT P95 变化 | 判断 |
| --- | ---: | ---: | --- |
| 合并上一轮提交与下一轮准备 | {state_tps_change:+.2f}% | {state_ttft_change:+.2f}% | 收益接近噪声 |
| 私有常驻 ExecutionLane | {lane_tps_change:+.2f}% | {lane_ttft_change:+.2f}% | 派发边界下降，保留 |

两组都是独立 A/B，各自按 `A/B/B/A/A/B` 交替运行。不能把两组绝对吞吐简单相减，因为 GPU 频率会漂移。

## 剩余 Driver 时间

连续热路径（相邻执行边界总 gap `<5 ms`）的 `driver_control` 进一步拆成：输出处理
均值 `{output_processing:.3f}`、转入原子推进 P50 `{transition:.3f}`、原子 apply/schedule/build
`{advance:.3f}`、进入下一次执行 `{next_entry:.3f} ms/step`。其中 apply、schedule 和 build 依赖
上一 token 的 CPU 结果，不能靠再次移动锁或再加一个线程安全隐藏。

若要继续接近 vLLM/SGLang，需要单独设计 GPU-side sampled-token relay、持久 batch/metadata 和
有界异步调度，让 CPU 准备 N+1 时不等待 N 的 token 回传。那是新的执行契约，不应伪装成本轮小修。

## 实验边界

- Qwen2.5-Coder-7B-Instruct、BF16、RTX 5090、32K KV tokens、block 16、
  max sequences 16、token budget 512。
- ShareGPT replay，Poisson 2/4/6/8 req/s；prefix cache、TTFT admission、speculation 关闭。
- 每点 64 请求 warmup 后 3 轮正式测试；吞吐取三轮中位数，TTFT 合并 192 个成功请求。
- Light strict 保持 completion claim，全部正式轮次要求零拒绝、零 self-resubmit。
- vLLM default 只作为 graph 性能上限；主归因基准是 vLLM eager。
- 36 个正式结果均为 64/64 请求成功，且每轮都生成相同的 7514 个输出 token；这是等工作量对照。
  BF16 动态批次会改变舍入路径，所以不同运行之间不保证生成 token ID 逐项完全相同，本文不作这项声明。

## 代码版本

- 原始基线：`{provenance["base_sha"]}`
- 合并状态推进：`{provenance["state_machine_sha"]}`
- 私有执行 lane：`{provenance["lane_sha"]}`
- 最终实验与分析：`{provenance["analysis_sha"]}`

## 图

- `figures/01_ttft_distribution.png`
- `figures/02_throughput_vs_eager.png`
- `figures/03_isolated_driver_ab.png`
- `figures/04_driver_gap_breakdown.png`
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Driver step-pipeline experiment.")
    parser.add_argument("input_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.input_root.resolve()
    output = (args.output_dir or root / "analysis").resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    matrix = _read_matrix(root / "final-matrix")
    isolated_ab = {
        "coalesced_state_machine": _read_ab(root / "state-machine-ab"),
        "private_execution_lane": _read_ab(root / "lane-ab"),
    }
    profiles = {
        "original": _read_profile(root / "profiles" / "original.json"),
        "coalesced state": _read_profile(root / "profiles" / "coalesced-state.json"),
        "private lane": _read_profile(root / "profiles" / "private-lane-control.json"),
    }
    provenance = _load(root / "DELIVERY.json")
    summary = {
        "scope": {
            "formal_runs_per_point": 3,
            "warmups_included": False,
            "ttft_aggregation": "pooled successful requests across three formal runs",
            "throughput_aggregation": "median of three formal runs",
        },
        "matrix": matrix,
        "isolated_ab": isolated_ab,
        "profiles": profiles,
        "provenance": provenance,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(matrix, output / "matrix.csv")
    _plot_ttft(matrix, figures / "01_ttft_distribution.png")
    _plot_throughput(matrix, figures / "02_throughput_vs_eager.png")
    _plot_ab(isolated_ab, figures / "03_isolated_driver_ab.png")
    _plot_gap(profiles, figures / "04_driver_gap_breakdown.png")
    _write_report(summary, output / "report.md")
    print(output)


if __name__ == "__main__":
    main()
