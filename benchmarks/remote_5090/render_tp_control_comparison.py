from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LABELS = ("light-vllm\nGloo 基线", "light-vllm\nSocket 优化", "vLLM 0.26.0\neager")
COLORS = ("#94A3B8", "#0EA5E9", "#1E293B")
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


def _plot_runs(axis: Any, x: int, values: list[float]) -> None:
    """用三个小点保留逐轮波动，避免柱状图隐藏原始样本。"""

    for offset, value in zip(RUN_OFFSETS, values, strict=True):
        axis.scatter(
            x + offset,
            value,
            s=32,
            color="white",
            edgecolor="#334155",
            linewidth=1.0,
            zorder=4,
        )


def _style_axis(axis: Any) -> None:
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def render(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    records = tuple(
        summary["tp2"].get(name) for name in ("light_gloo", "light_socket", "vllm_eager")
    )
    if any(record is None for record in records):
        raise ValueError("summary is missing one or more TP=2 comparison records")

    throughput = [float(record["throughput_median"]) for record in records]
    throughput_runs = [record["throughput_runs"] for record in records]
    tpot = [float(record["tpot_p50_ms_median"]) for record in records]
    tpot_runs = [record["tpot_p50_ms_runs"] for record in records]
    target = throughput[-1] * 0.92

    _configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 7.0))
    figure.subplots_adjust(left=0.065, right=0.98, bottom=0.23, top=0.78, wspace=0.18)
    figure.patch.set_facecolor("#F8FAFC")
    for axis in axes:
        axis.set_facecolor("white")

    bars = axes[0].bar(range(3), throughput, width=0.62, color=COLORS, zorder=2)
    for index, values in enumerate(throughput_runs):
        _plot_runs(axes[0], index, values)
    axes[0].axhspan(target, throughput[-1], color="#22C55E", alpha=0.07, zorder=0)
    axes[0].axhline(target, color="#16A34A", linestyle="--", linewidth=1.2)
    axes[0].text(
        2.42,
        target + 8,
        f"8% 差距目标  {target:.1f}",
        color="#15803D",
        fontsize=9,
        ha="right",
    )
    axes[0].bar_label(bars, labels=[f"{value:.1f}" for value in throughput], padding=7, fontsize=11)
    axes[0].set_xticks(range(3), LABELS, fontsize=10)
    axes[0].set_ylim(0, 1450)
    axes[0].set_ylabel("输出吞吐（token/s）")
    axes[0].set_title("吞吐：越高越好", loc="left", pad=14)
    _style_axis(axes[0])

    bars = axes[1].bar(range(3), tpot, width=0.62, color=COLORS, zorder=2)
    for index, values in enumerate(tpot_runs):
        _plot_runs(axes[1], index, values)
    axes[1].bar_label(bars, labels=[f"{value:.3f}" for value in tpot], padding=7, fontsize=11)
    axes[1].set_xticks(range(3), LABELS, fontsize=10)
    axes[1].set_ylim(0, 16)
    axes[1].set_ylabel("TPOT P50（ms/token）")
    axes[1].set_title("逐 token 延迟：越低越好", loc="left", pad=14)
    _style_axis(axes[1])

    gap = float(summary["tp2"]["socket_vs_vllm_gap_percent"])
    speedup = float(summary["tp2"]["socket_vs_gloo_throughput_percent"])
    figure.suptitle(
        "light-vllm TP=2 控制路径优化 vs vLLM",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
        y=0.965,
    )
    figure.text(
        0.5,
        0.905,
        f"Socket 相对 Gloo 吞吐 +{speedup:.2f}% · 达到 vLLM 的 {100 - gap:.2f}% · 差距 {gap:.2f}%",
        ha="center",
        fontsize=11,
        color="#0369A1",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.025,
        "Qwen2.5-Coder-7B-Instruct · BF16 · 2×RTX 5090（SYS，无 P2P） · 固定 16×512 decode\n"
        "柱为 3 次正式运行中位数，圆点为逐轮结果；warmup 不计。vLLM 使用 eager，关闭 CUDA Graph。",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#64748B",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor=figure.get_facecolor())
    figure.savefig(output.with_suffix(".svg"), facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render(summary, args.output)


if __name__ == "__main__":
    main()
