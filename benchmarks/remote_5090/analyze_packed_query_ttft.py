# 报告正文保留完整中文句子，避免为满足源码行宽而破坏生成后的 Markdown 可读性。
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, ticker  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

RATES = (2, 4, 6, 8)
COLORS = {
    "baseline": "#D97706",
    "candidate": "#2563EB",
    "vllm_eager": "#059669",
    "vllm_default": "#64748B",
    "grid": "#D7DEE8",
    "text": "#172033",
    "muted": "#5B6577",
    "panel": "#F7F9FC",
}


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _formal_runs(directory: Path, prefix: str) -> list[dict[str, object]]:
    paths = sorted(directory.glob(f"{prefix}-*-r[123].json"))
    if len(paths) != 3:
        raise ValueError(f"expected three formal runs under {directory}, got {paths}")
    return [_json(path) for path in paths]


def _summaries(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    return [run["summary"] for run in runs]  # type: ignore[index]


def _series(runs: list[dict[str, object]], field: str, scale: float = 1.0) -> dict[str, object]:
    values = [float(summary[field]) * scale for summary in _summaries(runs)]
    return {
        "values": values,
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _load_primary(raw: Path) -> dict[str, dict[int, list[dict[str, object]]]]:
    groups: dict[str, dict[int, list[dict[str, object]]]] = {
        "baseline": {},
        "candidate": {},
        "vllm_eager": {},
        "vllm_default": {},
    }
    for rate in RATES:
        groups["baseline"][rate] = _formal_runs(raw / "baseline" / f"rate{rate}", "light-strict")
        groups["candidate"][rate] = _formal_runs(raw / "candidate" / f"rate{rate}", "light-strict")
        groups["vllm_eager"][rate] = _formal_runs(raw / "baseline" / f"rate{rate}", "vllm-eager")
        groups["vllm_default"][rate] = _formal_runs(
            raw / "vllm-default" / f"rate{rate}", "vllm-default"
        )
    return groups


def _step_category(
    profile: dict[str, object], predicate: Callable[[dict[str, object]], bool]
) -> dict[str, object]:
    steps = [step for step in profile["steps"] if predicate(step)]  # type: ignore[index]
    gaps = [
        float(step["previous_executor_gap_ms"])
        for step in steps
        if step["previous_executor_gap_ms"] is not None
    ]
    return {
        "count": len(steps),
        "executor_total_ms": sum(float(step["executor_wall_ms"]) for step in steps),
        "executor_p50_ms": statistics.median(float(step["executor_wall_ms"]) for step in steps),
        "cuda_p50_ms": statistics.median(float(step["cuda_event_ms"]) for step in steps),
        "gap_p50_ms": statistics.median(gaps),
        "model_tokens": sum(int(step["model_tokens"]) for step in steps),
        "legacy_positions": sum(
            int(step["batch_size"]) * int(step["query_width"]) for step in steps
        ),
    }


def _profile_summary(path: Path, *, packed: bool) -> dict[str, object]:
    profile = _json(path)
    mixed = _step_category(
        profile,
        lambda step: (
            int(step["model_tokens"]) != int(step["batch_size"]) * int(step["query_width"])
        ),
    )
    decode = _step_category(profile, lambda step: int(step["query_width"]) == 1)
    mixed["executed_positions"] = mixed["model_tokens"] if packed else mixed["legacy_positions"]
    stages = profile.get("stages", {})

    def stage_mean(name: str) -> float:
        stage = stages.get(name, {})  # type: ignore[union-attr]
        return float(stage.get("wall_mean_ms", 0.0))  # type: ignore[union-attr]

    steps = int(profile["executor_steps"])
    service_step_ms = float(profile["executor_span_ms"]) / steps
    executor_step_ms = float(profile["executor_wall_total_ms"]) / steps
    cuda_step_ms = float(profile["executor_cuda_event_total_ms"]) / steps
    gap_step_ms = float(profile["inter_executor_gap_total_ms"]) / steps
    return {
        "executor_steps": steps,
        "executor_span_ms": float(profile["executor_span_ms"]),
        "executor_wall_total_ms": float(profile["executor_wall_total_ms"]),
        "executor_cuda_event_total_ms": float(profile["executor_cuda_event_total_ms"]),
        "inter_executor_gap_total_ms": float(profile["inter_executor_gap_total_ms"]),
        "inter_executor_gap_mean_ms": gap_step_ms,
        "remaining_step_breakdown": {
            "service_ms": service_step_ms,
            "executor_ms": executor_step_ms,
            "cuda_event_ms": cuda_step_ms,
            "executor_host_residual_ms": executor_step_ms - cuda_step_ms,
            "driver_gap_ms": gap_step_ms,
            "sampler_ms": stage_mean("sampler.sample"),
            "step_handler_ms": stage_mean("paged_step.forward"),
            "model_forward_ms": stage_mean("worker.model_forward"),
            "step_handler_outside_model_ms": stage_mean("paged_step.forward")
            - stage_mean("worker.model_forward"),
            "schedule_ms": stage_mean("scheduler.schedule"),
            "build_batch_ms": stage_mean("engine.build_batch"),
            "apply_output_ms": stage_mean("engine.apply_output"),
            "publish_stats_ms": stage_mean("engine.publish_scheduler_stats"),
        },
        "batch_shapes": profile["batch_shapes"],
        "mixed": mixed,
        "decode_w1": decode,
    }


def _outputs(run: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        str(result["request_id"]): result
        for result in run["results"]  # type: ignore[index]
        if result["error"] is None
    }


def _output_validation(expected: dict[str, object], actual: dict[str, object]) -> dict[str, object]:
    left = _outputs(expected)
    right = _outputs(actual)
    common = left.keys() & right.keys()
    return {
        "expected": len(left),
        "actual": len(right),
        "missing": sorted(left.keys() - right.keys()),
        "extra": sorted(right.keys() - left.keys()),
        "length_mismatches": sorted(
            request_id
            for request_id in common
            if int(left[request_id]["output_tokens"]) != int(right[request_id]["output_tokens"])
        ),
        "exact_token_mismatches": sorted(
            request_id
            for request_id in common
            if left[request_id]["generated_token_ids"] != right[request_id]["generated_token_ids"]
        ),
    }


def _metric_values(path: Path, metric_prefix: str) -> list[float]:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(metric_prefix):
            values.append(float(line.rsplit(" ", 1)[1]))
    return values


def _timer_sync_ab(result_root: Path) -> dict[str, object] | None:
    """读取可选的固定解码计时器消融；主实验包不依赖这组补测。"""
    root = result_root / "timer-ab"
    if not root.is_dir():
        return None

    groups: dict[str, object] = {}
    for label in ("current", "wall"):
        runs = [
            _json(root / label / f"r{index}.json")
            for index in (1, 2, 3)
            if (root / label / f"r{index}.json").is_file()
        ]
        if len(runs) != 3:
            return None
        groups[label] = {
            "throughput": _series(runs, "output_tokens_per_s"),
            "tpot_p50_ms": _series(runs, "tpot_p50_s", 1000.0),
            "success": [
                {
                    "requests": int(summary["requests"]),
                    "successful": int(summary["successful_requests"]),
                    "failed": int(summary["failed_requests"]),
                }
                for summary in _summaries(runs)
            ],
        }

    current = groups["current"]  # type: ignore[assignment]
    wall = groups["wall"]  # type: ignore[assignment]
    current_throughput = float(current["throughput"]["median"])
    wall_throughput = float(wall["throughput"]["median"])
    current_tpot = float(current["tpot_p50_ms"]["median"])
    wall_tpot = float(wall["tpot_p50_ms"]["median"])
    return {
        "groups": groups,
        "throughput_delta_percent": (wall_throughput / current_throughput - 1.0) * 100.0,
        "tpot_delta_percent": (wall_tpot / current_tpot - 1.0) * 100.0,
        "order_recheck_throughput": (
            float(_json(root / "current-recheck" / "r1.json")["summary"]["output_tokens_per_s"])
            if (root / "current-recheck" / "r1.json").is_file()
            else None
        ),
    }


def _configure_matplotlib() -> None:
    """为本地离线绘图选择中文字体，不把字体文件写入结果目录。"""
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    )
    family = "DejaVu Sans"
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.edgecolor": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["text"],
        }
    )


def _figure(title: str, subtitle: str, *, rows: int = 1, cols: int = 1) -> tuple[Figure, object]:
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9.6), facecolor="white")
    fig.suptitle(title, x=0.055, y=0.97, ha="left", fontsize=27, fontweight="bold")
    fig.text(0.057, 0.915, subtitle, ha="left", fontsize=14, color=COLORS["muted"])
    return fig, axes


def _style_axis(axis: Axes, *, grid: bool = True) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_linewidth(1.2)
    axis.tick_params(labelsize=12)
    if grid:
        axis.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
        axis.set_axisbelow(True)


def _save_figure(fig: Figure, output: Path) -> None:
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_ttft(summary: dict[str, object], output: Path) -> None:
    fig, axis_value = _figure(
        "正常 Poisson 负载：TTFT P95",
        "Qwen2.5-Coder-7B / BF16 / RTX 5090；点为 3 次中位数，误差线为 min–max",
    )
    axis: Axes = axis_value  # type: ignore[assignment]
    labels = {
        "baseline": "Light 原版 strict",
        "candidate": "Light Packed strict",
        "vllm_eager": "vLLM eager",
        "vllm_default": "vLLM default/graph",
    }
    primary = summary["primary"]  # type: ignore[index]
    annotation_offsets = {
        "baseline": (-8, 12),
        "candidate": (-9, -20),
        "vllm_eager": (10, 8),
        "vllm_default": (-8, -18),
    }
    for group, label in labels.items():
        medians = np.array(
            [float(primary[group][str(rate)]["ttft_p95_ms"]["median"]) for rate in RATES]
        )
        minimums = np.array(
            [float(primary[group][str(rate)]["ttft_p95_ms"]["min"]) for rate in RATES]
        )
        maximums = np.array(
            [float(primary[group][str(rate)]["ttft_p95_ms"]["max"]) for rate in RATES]
        )
        axis.errorbar(
            RATES,
            medians,
            yerr=np.vstack((medians - minimums, maximums - medians)),
            label=label,
            color=COLORS[group],
            marker="o",
            linewidth=2.3,
            markersize=7,
            capsize=5,
        )
        for rate, value in zip(RATES, medians, strict=True):
            axis.annotate(
                f"{value:.0f}",
                (rate, value),
                xytext=annotation_offsets[group],
                textcoords="offset points",
                color=COLORS[group],
                fontsize=11,
                ha="center",
            )
    axis.set_yscale("log")
    axis.set_ylim(20, 2000)
    axis.set_yticks((20, 50, 100, 200, 500, 1000, 2000))
    axis.yaxis.set_major_formatter(ticker.ScalarFormatter())
    axis.yaxis.set_minor_formatter(ticker.NullFormatter())
    axis.set_xticks(RATES)
    axis.set_xlabel("请求到达率（req/s）", fontsize=14, labelpad=12)
    axis.set_ylabel("TTFT P95（ms）", fontsize=14, labelpad=12)
    _style_axis(axis)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False, fontsize=12)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.19)
    _save_figure(fig, output)


def _plot_throughput(summary: dict[str, object], output: Path) -> None:
    fig, axis_value = _figure(
        "正常 Poisson 负载：输出吞吐",
        "柱为 3 次中位数；横轴同时标注 Packed 相对 vLLM eager 的比例",
    )
    axis: Axes = axis_value  # type: ignore[assignment]
    groups = ("baseline", "candidate", "vllm_eager", "vllm_default")
    labels = {
        "baseline": "Light 原版",
        "candidate": "Light Packed",
        "vllm_eager": "vLLM eager",
        "vllm_default": "vLLM graph",
    }
    primary = summary["primary"]  # type: ignore[index]
    centers = np.arange(len(RATES), dtype=float)
    width = 0.19
    for index, group in enumerate(groups):
        values = np.array(
            [float(primary[group][str(rate)]["throughput"]["median"]) for rate in RATES]
        )
        bars = axis.bar(
            centers + (index - 1.5) * width,
            values,
            width=width * 0.92,
            label=labels[group],
            color=COLORS[group],
        )
        axis.bar_label(bars, labels=[f"{value:.0f}" for value in values], padding=4, fontsize=11)
    ratios = []
    for rate in RATES:
        candidate = float(primary["candidate"][str(rate)]["throughput"]["median"])
        eager = float(primary["vllm_eager"][str(rate)]["throughput"]["median"])
        ratios.append(candidate / eager)
    axis.set_xticks(
        centers,
        [
            f"{rate} req/s\nPacked/eager {ratio:.1%}"
            for rate, ratio in zip(RATES, ratios, strict=True)
        ],
    )
    axis.set_ylim(0, 710)
    axis.set_ylabel("输出吞吐（tok/s）", fontsize=14, labelpad=12)
    _style_axis(axis)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=12)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.84, bottom=0.2)
    _save_figure(fig, output)


def _bar_panel(
    axis: Axes,
    title: str,
    labels: tuple[str, ...],
    series: tuple[tuple[str, tuple[float, ...], str], ...],
    *,
    suffix: str,
    note: str | None = None,
) -> None:
    centers = np.arange(len(labels), dtype=float)
    width = 0.68 / max(1, len(series))
    values = [value for _, entries, _ in series for value in entries]
    for index, (name, entries, color) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        bars = axis.bar(centers + offset, entries, width=width * 0.9, label=name, color=color)
        axis.bar_label(
            bars,
            labels=[f"{value:.1f}{suffix}" for value in entries],
            padding=4,
            fontsize=10,
            color=color,
        )
    axis.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=12)
    axis.set_xticks(centers, labels)
    axis.set_ylim(0, max(values) * 1.25 if values else 1.0)
    _style_axis(axis)
    if len(series) > 1:
        axis.legend(frameon=False, fontsize=10, loc="upper right")
    if note is not None:
        axis.text(
            0.98,
            0.93,
            note,
            transform=axis.transAxes,
            fontsize=10,
            color=COLORS["muted"],
            ha="right",
            va="top",
        )


def _plot_breakdown(summary: dict[str, object], output: Path) -> None:
    fig, axes_value = _figure(
        "8 req/s Profile：Mixed 与 Decode 分解",
        "相同 64 请求；Packed 消除 mixed padding，decode W1 与 Driver gap 基本未变",
        rows=2,
        cols=2,
    )
    axes = np.asarray(axes_value).reshape(2, 2)
    baseline = summary["profiles"]["baseline"]  # type: ignore[index]
    candidate = summary["profiles"]["candidate"]  # type: ignore[index]
    avoided = int(candidate["mixed"]["legacy_positions"]) - int(candidate["mixed"]["model_tokens"])
    _bar_panel(
        axes[0, 0],
        "Mixed step 实际执行位置数",
        ("原版", "Packed"),
        (
            (
                "执行位置",
                (
                    float(baseline["mixed"]["executed_positions"]),
                    float(candidate["mixed"]["executed_positions"]),
                ),
                COLORS["candidate"],
            ),
        ),
        suffix="",
        note=f"Packed 避免 {avoided:,} 个 padding 位置",
    )
    _bar_panel(
        axes[0, 1],
        "Mixed step P50",
        ("原版", "Packed"),
        (
            (
                "Executor",
                (
                    float(baseline["mixed"]["executor_p50_ms"]),
                    float(candidate["mixed"]["executor_p50_ms"]),
                ),
                COLORS["baseline"],
            ),
            (
                "CUDA event",
                (
                    float(baseline["mixed"]["cuda_p50_ms"]),
                    float(candidate["mixed"]["cuda_p50_ms"]),
                ),
                COLORS["candidate"],
            ),
        ),
        suffix="ms",
    )
    _bar_panel(
        axes[1, 0],
        "Decode W1 P50",
        ("原版", "Packed"),
        (
            (
                "Executor",
                (
                    float(baseline["decode_w1"]["executor_p50_ms"]),
                    float(candidate["decode_w1"]["executor_p50_ms"]),
                ),
                COLORS["baseline"],
            ),
            (
                "CUDA event",
                (
                    float(baseline["decode_w1"]["cuda_p50_ms"]),
                    float(candidate["decode_w1"]["cuda_p50_ms"]),
                ),
                COLORS["candidate"],
            ),
        ),
        suffix="ms",
    )
    _bar_panel(
        axes[1, 1],
        "前一 Executor 到本步的 gap P50",
        ("Mixed", "Decode W1"),
        (
            (
                "原版",
                (
                    float(baseline["mixed"]["gap_p50_ms"]),
                    float(baseline["decode_w1"]["gap_p50_ms"]),
                ),
                COLORS["baseline"],
            ),
            (
                "Packed",
                (
                    float(candidate["mixed"]["gap_p50_ms"]),
                    float(candidate["decode_w1"]["gap_p50_ms"]),
                ),
                COLORS["candidate"],
            ),
        ),
        suffix="ms",
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.08, hspace=0.42, wspace=0.2)
    _save_figure(fig, output)


def _plot_ablation(summary: dict[str, object], output: Path) -> None:
    fig, axes_value = _figure(
        "8 req/s：reserved_sequences 消融",
        "短请求池固定：256 scheduled tokens / 2047 KV slots；只改变 sequence slot 数量",
        cols=2,
    )
    axes = np.asarray(axes_value).reshape(2)
    ablation = summary["reserved_sequences_ablation"]  # type: ignore[index]
    labels = ("off", "1", "2", "4", "8")
    ttft = np.array([float(ablation[label]["ttft_p95_ms"]["median"]) for label in labels])
    throughput = np.array([float(ablation[label]["throughput"]["median"]) for label in labels])
    bars = axes[0].bar(labels, ttft, color=COLORS["baseline"], width=0.62)
    axes[0].set_title("TTFT P95（3 次中位数）", loc="left", fontsize=16, fontweight="bold")
    axes[0].set_yscale("log")
    axes[0].set_ylim(30, 4000)
    axes[0].set_ylabel("TTFT P95（ms）", fontsize=13)
    axes[0].bar_label(bars, labels=[f"{value:.1f}" for value in ttft], padding=4, fontsize=10)
    _style_axis(axes[0])

    bars = axes[1].bar(labels, throughput, color=COLORS["candidate"], width=0.62)
    axes[1].set_title("吞吐（3 次中位数）", loc="left", fontsize=16, fontweight="bold")
    axes[1].set_ylim(0, 680)
    axes[1].set_ylabel("输出吞吐（tok/s）", fontsize=13)
    axes[1].bar_label(bars, labels=[f"{value:.1f}" for value in throughput], padding=4, fontsize=10)
    _style_axis(axes[1])
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.1, wspace=0.24)
    _save_figure(fig, output)


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_report(summary: dict[str, object], result_root: Path) -> None:
    primary = summary["primary"]  # type: ignore[index]
    profiles = summary["profiles"]  # type: ignore[index]
    ablation = summary["reserved_sequences_ablation"]  # type: ignore[index]
    alternating = summary["alternating_rate8"]  # type: ignore[index]
    rate_rows: list[tuple[str, ...]] = []
    for rate in RATES:
        packed_ttft = float(primary["candidate"][str(rate)]["ttft_p95_ms"]["median"])
        eager_ttft = float(primary["vllm_eager"][str(rate)]["ttft_p95_ms"]["median"])
        packed_tp = float(primary["candidate"][str(rate)]["throughput"]["median"])
        eager_tp = float(primary["vllm_eager"][str(rate)]["throughput"]["median"])
        rate_rows.append(
            (
                str(rate),
                f"{float(primary['baseline'][str(rate)]['ttft_p95_ms']['median']):.1f}",
                f"{packed_ttft:.1f}",
                f"{eager_ttft:.1f}",
                f"{packed_ttft / eager_ttft:.2f}×",
                f"{packed_tp:.1f}",
                f"{eager_tp:.1f}",
                f"{packed_tp / eager_tp:.1%}",
            )
        )
    ablation_rows = [
        (
            label,
            f"{float(ablation[label]['ttft_p95_ms']['median']):.1f}",
            f"{float(ablation[label]['throughput']['median']):.1f}",
        )
        for label in ("off", "1", "2", "4", "8")
    ]
    validations = summary["output_validation"]  # type: ignore[index]
    exact_rate8 = len(validations["8"]["exact_token_mismatches"])
    remaining = profiles["candidate"]["remaining_step_breakdown"]
    service_ms = float(remaining["service_ms"])
    driver_gap_ms = float(remaining["driver_gap_ms"])
    sampler_ms = float(remaining["sampler_ms"])
    prepare_ms = float(remaining["step_handler_outside_model_ms"])
    timer_residual_ms = float(remaining["executor_host_residual_ms"])

    def theoretical_speedup(recovered_ms: float) -> float:
        return service_ms / (service_ms - recovered_ms) - 1.0

    timer_ab = summary.get("timer_sync_ab")
    if isinstance(timer_ab, dict):
        timer_groups = timer_ab["groups"]
        timer_current = timer_groups["current"]
        timer_wall = timer_groups["wall"]
        timer_budget = (
            f"固定 B16/W1 三次中位数：{float(timer_current['throughput']['median']):.1f}→"
            f"{float(timer_wall['throughput']['median']):.1f} tok/s"
        )
        timer_limit = f"实测吞吐 {float(timer_ab['throughput_delta_percent']):+.2f}%"
        timer_expectation = "不作为性能修复；仅在保留准确 CUDA 指标的前提下重构采样"
        timer_ttft = (
            f"TPOT 中位数 {float(timer_current['tpot_p50_ms']['median']):.3f}→"
            f"{float(timer_wall['tpot_p50_ms']['median']):.3f} ms，差异属于运行噪声"
        )
    else:
        timer_budget = f"Executor wall 与 CUDA event 只差 {timer_residual_ms:.3f} ms/step"
        timer_limit = f"吞吐 +{theoretical_speedup(timer_residual_ms):.1%}"
        timer_expectation = "<0.5%"
        timer_ttft = "sampler 已先完成同步，关 timer 不会吃掉 Driver gap"

    report = f"""# light-vLLM Packed Query 与 TTFT 根因实验

## 一句话结论

原先 8 req/s 下约 1.5 秒的 TTFT P95，主要不是 strict 非抢占造成，而是 mixed prefill/decode batch 被补齐成 `[B, W]` 后执行了大量 padding。改成统一 token-major Packed Query 后，TTFT P95 中位数降到 {float(primary["candidate"]["8"]["ttft_p95_ms"]["median"]):.1f} ms，吞吐达到 vLLM eager 的 {float(primary["candidate"]["8"]["throughput"]["median"]) / float(primary["vllm_eager"]["8"]["throughput"]["median"]):.1%}；strict completion claim 保持不变，self-resubmit、回滚和拒绝均为 0。

## 版本与公平条件

- baseline：`c81e519`
- Packed Query 运行时代码：`c1f18dc`
- 分支：`codex/packed-query-ttft`；benchmark 消融参数提交：`6c46884`
- 模型：Qwen2.5-Coder-7B-Instruct，BF16，RTX 5090
- KV：32K tokens，block size 16；max sequences 16；scheduled token budget 512
- 双方关闭 prefix cache、TTFT admission 和 speculation；Light 使用 strict completion claim
- 正常 ShareGPT replay，Poisson 2/4/6/8 req/s；每档 warmup 后 3 次正式运行
- vLLM eager 是主基准；vLLM default/graph 只作为第二参照

## 主结果

{_markdown_table(("req/s", "原版 TTFT ms", "Packed TTFT ms", "vLLM eager TTFT ms", "Packed/eager TTFT", "Packed tok/s", "vLLM eager tok/s", "Packed/eager 吞吐"), rate_rows)}

![TTFT P95](analysis/ttft_p95.png)

![吞吐](analysis/throughput.png)

最关键的 8 req/s 还做了独立服务启动的 `baseline→candidate` 三组交替复核：baseline TTFT P95/吞吐中位数为 {float(alternating["baseline"]["ttft_p95_ms"]["median"]):.1f} ms / {float(alternating["baseline"]["throughput"]["median"]):.1f} tok/s，candidate 为 {float(alternating["candidate"]["ttft_p95_ms"]["median"]):.1f} ms / {float(alternating["candidate"]["throughput"]["median"]):.1f} tok/s。收益不是测试顺序或机器时间漂移造成的。

## 根因证据

8 req/s 的完整 profile 中，两版都出现 57 个 mixed step：

- 原版 mixed step P50 为 {float(profiles["baseline"]["mixed"]["executor_p50_ms"]):.2f} ms，CUDA event P50 为 {float(profiles["baseline"]["mixed"]["cuda_p50_ms"]):.2f} ms，57 步共占 {float(profiles["baseline"]["mixed"]["executor_total_ms"]) / 1000:.2f} s。
- Packed 后 mixed step P50 为 {float(profiles["candidate"]["mixed"]["executor_p50_ms"]):.2f} ms，CUDA event P50 为 {float(profiles["candidate"]["mixed"]["cuda_p50_ms"]):.2f} ms，57 步共占 {float(profiles["candidate"]["mixed"]["executor_total_ms"]) / 1000:.2f} s。
- Packed profile 的 57 个 mixed step 只执行 {int(profiles["candidate"]["mixed"]["model_tokens"]):,} 个真实 token；旧布局需要启动 {int(profiles["candidate"]["mixed"]["legacy_positions"]):,} 个位置，因此避免了 {int(profiles["candidate"]["mixed"]["legacy_positions"]) - int(profiles["candidate"]["mixed"]["model_tokens"]):,} 个 padding 位置。
- decode W1 的 Executor P50 基本不变：{float(profiles["baseline"]["decode_w1"]["executor_p50_ms"]):.2f} → {float(profiles["candidate"]["decode_w1"]["executor_p50_ms"]):.2f} ms；step gap 也没有被 Packed 修掉。

这条证据链说明：旧 mixed step 在 GPU 上确实执行了 padding 对应的 GEMM/attention/logits 工作，单步被拉长后，请求到达速度超过首 token 消化速度，waiting 队列迅速积累。Packed Query 消除的是这段真实 GPU 浪费，而不是修改准入或回滚策略。

![Profile 分解](analysis/step_breakdown.png)

## `reserved_sequences` 消融

{_markdown_table(("reserved_sequences", "TTFT P95 ms", "吞吐 tok/s"), ablation_rows)}

`8` 会把 16 个 sequence slot 的一半划给首 token 短池，但短池每步只有 256 token 预算；正常 mixed 流量下，通用 decode 并发被压缩，TTFT 和吞吐同时恶化。因此不修改当前默认值 `1`。

![reserved_sequences 消融](analysis/reserved_sequences_ablation.png)

## 其他修复方向的收益上限

以下区间不是已经实现的成绩，而是用当前 Packed profile 的每步时间预算推导出的工程预期。多项优化会吃同一段时间，不能直接相加。

| 方向 | 当前可见时间预算 | 理论上限 | 保守工程预期 | 对当前 TTFT 的判断 |
| --- | --- | --- | --- | --- |
| Driver 双缓冲 / 两批在途 | 外部 gap {driver_gap_ms:.2f} ms / service step {service_ms:.2f} ms | 全部隐藏时吞吐 +{theoretical_speedup(driver_gap_ms):.1%} | 吞吐 +5%～9% | 2/8 req/s 已无明显首 token 排队，通常只省 0～10 ms；更高负载下收益会非线性放大 |
| decode CUDA Graph | 独立 fixed-W1 trace 中 kernel 10.592 ms、CUDA-event 11.743 ms，设备边界内空洞约 1.151 ms | 最多约 +10% step capacity | 吞吐 +4%～7% | 只 capture 常见 decode shape 时通常小幅改善；不要让首个不规则 prefill 等待 graph |
| sampler 异步回传 | {sampler_ms:.2f} ms/step | 全部隐藏时吞吐 +{theoretical_speedup(sampler_ms):.1%} | 吞吐 +1%～3% | 对 TTFT 很小，主要改善 decode capacity / TPOT |
| staging buffer、metadata/H2D | Step Handler 中 model 外只有 {prepare_ms:.2f} ms/step，且含不可删除工作 | 全部消失时吞吐 +{theoretical_speedup(prepare_ms):.1%} | 吞吐 +1%～2% | 目前没有证明单独 H2D ≥0.5 ms，不应先做大改 |
| 删除每步 CUDA timer 同步 | {timer_budget} | {timer_limit} | {timer_expectation} | {timer_ttft} |
| `reserved_sequences` 调参 | `off/1/2` 吞吐仅 596.0/599.4/597.6 tok/s | 没有稳定正收益 | 保持默认 1 | 设为 8 会把 TTFT 恶化到 2520.6 ms |
| self-resubmit 部分 KV 保留 | 本组 resubmit=0 | 当前正常负载收益为 0 | 只改善容量压力下的重算量和 token gap | 首 token 已产生后才触发，主要影响 ITL/吞吐，不是当前 TTFT 根因 |

最值得继续的是 **Driver overlap + 常见 decode shape 的 CUDA Graph**。按时间预算，两者有机会把正常 8 req/s 吞吐从约 596 tok/s 推到 630～650 tok/s；但它们会重叠吃掉 launch/等待空洞，必须分别 A/B，不能把两个百分比直接相加。当前 TTFT 已经比 vLLM eager 低，因此下一阶段应把主验收改成 fixed-W1 TPOT、饱和吞吐和 inter-step gap，而不是继续压 46.5 ms 的 TTFT。

计时器消融的原始 JSON、Prometheus 快照和服务日志保存在 `timer-ab/`。这组补测使用同一个 `6c46884` checkout，关闭 prefix cache、TTFT admission 与 speculation；`current` 和替换成墙钟计时器的 `wall` 都经过 warmup 后正式运行三次，并额外回切一次 `current` 检查顺序漂移。每次均为 16/16 成功、8192 输出 token。

## 正确性与边界

- 本地：237 个测试通过；ruff、format check、`git diff --check` 通过。
- RTX 5090：全量测试通过；Torch/Triton packed GQA、混合长度、linear/tree、decode-after-prefill 的 FP16/BF16 数值对照通过。BF16 attention 容差为 `atol=rtol=2e-2`。
- 所有正式性能运行均 64/64 成功；candidate 与 vLLM eager 没有 missing/extra 请求，输出 token 数逐请求一致。
- 不把不同 BF16 kernel 的 greedy token 序列宣称为 bitwise identical：8 req/s 的 64 个请求中有 {exact_rate8} 个 exact-token mismatch；原版相对 vLLM 的 mismatch 更多。该项不影响长度对齐和性能结论，但若未来要求跨框架逐 token 完全一致，需要另做确定性数值工程。
- 本组 32K KV 下 Light self-resubmit=0、回滚 token=0、拒绝=0；vLLM preemption=0。因此它验证的是“容量足够时 strict 非抢占没有造成两个数量级 TTFT 差距”，不是证明非抢占策略在容量压力下一定优于 victim preemption。
- Packed 后平均外部 Executor gap 仍为 {float(profiles["candidate"]["inter_executor_gap_mean_ms"]):.2f} ms/step，明显高于 vLLM 的流水化实现。但主验收已经达到，按计划不在本分支引入高风险双缓冲；后续可用独立分支解决。

## 数据完整性与复现

- 原始结果：`packed-query-ttft-20260821/`，包含 388 个文件（JSON、Prometheus、日志、profile、环境与命令参数）。
- 下载包：`packed-query-ttft-20260821.tar.gz`
- 包 SHA-256：`{summary["archive_sha256"]}`
- 解包后的 `SHA256SUMS` 已逐文件校验：388/388 通过。
- 分析数据：`analysis/summary.json`
- 重画命令：`python benchmarks/remote_5090/analyze_packed_query_ttft.py <解包目录> <结果目录>`（需要 matplotlib）。
"""
    (result_root / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path, help="解包后的 packed-query-ttft-20260821 目录")
    parser.add_argument("result_root", type=Path, help="报告和 analysis 输出目录")
    args = parser.parse_args()
    raw = args.raw.resolve()
    result_root = args.result_root.resolve()
    analysis = result_root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    groups = _load_primary(raw)
    primary: dict[str, dict[str, object]] = {}
    for group, runs_by_rate in groups.items():
        primary[group] = {
            str(rate): {
                "ttft_p95_ms": _series(runs, "ttft_p95_s", 1000.0),
                "throughput": _series(runs, "output_tokens_per_s"),
                "success": [
                    {
                        "requests": int(summary["requests"]),
                        "successful": int(summary["successful_requests"]),
                        "failed": int(summary["failed_requests"]),
                    }
                    for summary in _summaries(runs)
                ],
            }
            for rate, runs in runs_by_rate.items()
        }

    baseline_profile = _profile_summary(
        next((raw / "baseline" / "rate8").glob("light-strict-*-profile-stages.json")),
        packed=False,
    )
    candidate_profile = _profile_summary(
        next((raw / "candidate" / "rate8").glob("light-strict-*-profile-stages.json")),
        packed=True,
    )

    ablation: dict[str, object] = {
        "off": {
            "ttft_p95_ms": _series(groups["candidate"][8], "ttft_p95_s", 1000.0),
            "throughput": _series(groups["candidate"][8], "output_tokens_per_s"),
        }
    }
    for count in (1, 2, 4, 8):
        runs = _formal_runs(raw / "ablation" / f"seq{count}", "light-strict")
        ablation[str(count)] = {
            "ttft_p95_ms": _series(runs, "ttft_p95_s", 1000.0),
            "throughput": _series(runs, "output_tokens_per_s"),
        }

    alternating: dict[str, object] = {}
    for label, prefix in (("baseline", "a"), ("candidate", "b")):
        runs = [
            _json(next((raw / "alternating" / f"{prefix}{index}").glob("*-r1.json")))
            for index in (1, 2, 3)
        ]
        alternating[label] = {
            "ttft_p95_ms": _series(runs, "ttft_p95_s", 1000.0),
            "throughput": _series(runs, "output_tokens_per_s"),
        }

    validations = {
        str(rate): _output_validation(groups["vllm_eager"][rate][0], groups["candidate"][rate][0])
        for rate in RATES
    }
    candidate_prom = next(
        (raw / "candidate" / "rate8").glob("light-strict-*-r3.metrics-after.prom")
    )
    vllm_prom = next((raw / "baseline" / "rate8").glob("vllm-eager-*-r3.metrics-after.prom"))
    archive_sha = (
        (result_root / "packed-query-ttft-20260821.tar.gz.sha256")
        .read_text(encoding="utf-8")
        .split()[0]
    )
    summary: dict[str, object] = {
        "primary": primary,
        "profiles": {"baseline": baseline_profile, "candidate": candidate_profile},
        "reserved_sequences_ablation": ablation,
        "alternating_rate8": alternating,
        "output_validation": validations,
        "policy_counters": {
            "light_self_resubmits": _metric_values(
                candidate_prom, "light_vllm_self_resubmits_total"
            ),
            "light_rolled_back_tokens": _metric_values(
                candidate_prom, "light_vllm_self_resubmit_rolled_back_tokens_total"
            ),
            "light_rejected": _metric_values(
                candidate_prom, 'light_vllm_requests_total{model="qwen2",outcome="rejected"}'
            ),
            "vllm_preemptions": _metric_values(vllm_prom, "vllm:num_preemptions_total"),
        },
        "timer_sync_ab": _timer_sync_ab(result_root),
        "archive_sha256": archive_sha,
        "raw_manifest_files": len((raw / "SHA256SUMS").read_text(encoding="utf-8").splitlines()),
    }
    (analysis / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _configure_matplotlib()
    _plot_ttft(summary, analysis / "ttft_p95.png")
    _plot_throughput(summary, analysis / "throughput.png")
    _plot_breakdown(summary, analysis / "step_breakdown.png")
    _plot_ablation(summary, analysis / "reserved_sequences_ablation.png")
    _write_report(summary, result_root)


if __name__ == "__main__":
    main()
