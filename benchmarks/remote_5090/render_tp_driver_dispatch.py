from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LABELS = ("light-vllm\nLane 基线", "light-vllm\nInline 候选（拒绝）", "vLLM 0.26.0\neager")
COLORS = ("#64748B", "#F59E0B", "#0EA5E9")
RUN_OFFSETS = (-0.12, 0.0, 0.12)


def _configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.sans-serif": (
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "SimHei",
                "DejaVu Sans",
            ),
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )


def _style_axis(axis: Any) -> None:
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def _plot_runs(axis: Any, x: int, values: list[float]) -> None:
    for offset, value in zip(RUN_OFFSETS, values, strict=True):
        axis.scatter(
            x + offset,
            value,
            s=26,
            color="white",
            edgecolor="#334155",
            linewidth=0.9,
            zorder=4,
        )


def _plot_metric(
    axis: Any,
    records: tuple[dict[str, Any], ...],
    *,
    median_key: str,
    runs_key: str,
    title: str,
    ylabel: str,
    precision: int,
) -> None:
    values = [float(record[median_key]) for record in records]
    bars = axis.bar(range(3), values, width=0.62, color=COLORS, zorder=2)
    bars[1].set_hatch("///")
    bars[1].set_edgecolor("#B45309")
    for index, record in enumerate(records):
        _plot_runs(axis, index, record[runs_key])
    axis.bar_label(
        bars,
        labels=[f"{value:.{precision}f}" for value in values],
        padding=5,
        fontsize=9,
    )
    axis.set_xticks(range(3), LABELS, fontsize=8.5)
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", pad=10)
    _style_axis(axis)


def render(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    records = tuple(
        summary["tp2"][name] for name in ("light_lane", "light_inline_rejected", "vllm_eager")
    )
    _configure_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(13.6, 10.2))
    figure.subplots_adjust(
        left=0.075,
        right=0.98,
        bottom=0.16,
        top=0.82,
        hspace=0.48,
        wspace=0.2,
    )
    figure.patch.set_facecolor("#F8FAFC")
    for axis in axes.flat:
        axis.set_facecolor("white")

    _plot_metric(
        axes[0, 0],
        records,
        median_key="throughput_median",
        runs_key="throughput_runs",
        title="输出吞吐：越高越好",
        ylabel="token/s",
        precision=1,
    )
    _plot_metric(
        axes[0, 1],
        records,
        median_key="tpot_p50_ms_median",
        runs_key="tpot_p50_ms_runs",
        title="TPOT P50：越低越好",
        ylabel="ms/token",
        precision=3,
    )
    _plot_metric(
        axes[1, 0],
        records,
        median_key="ttft_p95_ms_median",
        runs_key="ttft_p95_ms_runs",
        title="TTFT P95：越低越好",
        ylabel="ms",
        precision=1,
    )
    _plot_metric(
        axes[1, 1],
        records,
        median_key="max_itl_p95_ms_median",
        runs_key="max_itl_p95_ms_runs",
        title="单请求最大 token 间隔 P95：越低越好",
        ylabel="ms",
        precision=1,
    )

    comparison = summary["comparison"]
    figure.suptitle(
        "TP rank 0 执行边界实验：吞吐追平，但尾延迟回退",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
        y=0.965,
    )
    figure.text(
        0.5,
        0.91,
        "Inline 相对 Lane：吞吐 "
        f"+{comparison['inline_vs_lane_throughput_percent']:.2f}% · "
        "TPOT "
        f"-{comparison['inline_vs_lane_tpot_reduction_percent']:.2f}% · "
        "最大 token 间隔 P95 "
        f"+{comparison['inline_vs_lane_max_itl_regression_percent']:.2f}%",
        ha="center",
        fontsize=11,
        color="#B45309",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.035,
        "Qwen2.5-Coder-7B-Instruct · BF16 · TP=2 · 2×RTX 5090（SYS，无 P2P） · 固定 16×512 decode\n"
        "柱为 3 次正式运行中位数，圆点为逐轮结果；warmup 不计。斜线柱为已撤回候选，不是当前实现。",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#64748B",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor=figure.get_facecolor())
    svg_output = output.with_suffix(".svg")
    figure.savefig(svg_output, facecolor=figure.get_facecolor())
    plt.close(figure)
    # Matplotlib 的 path 数据默认保留行尾空格；生成后规范化，避免污染 Git diff。
    svg_text = svg_output.read_text(encoding="utf-8")
    svg_output.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render(summary, args.output)


if __name__ == "__main__":
    main()
