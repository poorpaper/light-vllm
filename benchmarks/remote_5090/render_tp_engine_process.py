from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LABELS = ("light-vLLM\nEngine/HTTP 分进程", "vLLM 0.26.0\neager")
COLORS = ("#E45756", "#4C78A8")
RUN_OFFSETS = (-0.10, 0.0, 0.10)


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
    bars = axis.bar(range(2), values, width=0.58, color=COLORS, zorder=2)
    for index, record in enumerate(records):
        for offset, value in zip(RUN_OFFSETS, record[runs_key], strict=True):
            axis.scatter(
                index + offset,
                value,
                s=28,
                color="white",
                edgecolor="#334155",
                linewidth=0.9,
                zorder=4,
            )
    median_labels = axis.bar_label(
        bars,
        labels=[f"{value:.{precision}f}" for value in values],
        padding=5,
        fontsize=9,
    )
    for label in median_labels:
        label.set_zorder(5)
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1})
    axis.set_xticks(range(2), LABELS, fontsize=8.5)
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", pad=10)
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def render(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    records = tuple(summary["tp2"][name] for name in ("light_engine_process", "vllm_eager"))
    _configure_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 9.6))
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.16,
        top=0.81,
        hspace=0.48,
        wspace=0.22,
    )
    figure.patch.set_facecolor("#F8FAFC")
    for axis in axes.flat:
        axis.set_facecolor("white")

    for axis, median_key, runs_key, title, ylabel, precision in (
        (
            axes[0, 0],
            "throughput_median",
            "throughput_runs",
            "输出吞吐：越高越好",
            "token/s",
            1,
        ),
        (
            axes[0, 1],
            "tpot_p50_ms_median",
            "tpot_p50_ms_runs",
            "TPOT P50：越低越好",
            "ms/token",
            3,
        ),
        (
            axes[1, 0],
            "ttft_p95_ms_median",
            "ttft_p95_ms_runs",
            "TTFT P95：越低越好",
            "ms",
            1,
        ),
        (
            axes[1, 1],
            "max_itl_p95_ms_median",
            "max_itl_p95_ms_runs",
            "单请求最大 token 间隔 P95：越低越好",
            "ms",
            1,
        ),
    ):
        _plot_metric(
            axis,
            records,
            median_key=median_key,
            runs_key=runs_key,
            title=title,
            ylabel=ylabel,
            precision=precision,
        )

    comparison = summary["comparison"]
    figure.suptitle(
        "TP=2 独立 Engine 进程：吞吐与稳态 TPOT 追平 vLLM",
        fontsize=17,
        fontweight="bold",
        color="#0F172A",
        y=0.965,
    )
    figure.text(
        0.5,
        0.905,
        "吞吐 "
        f"{comparison['light_vs_vllm_throughput_delta_percent']:+.2f}% · "
        "TPOT "
        f"{comparison['light_vs_vllm_tpot_delta_percent']:+.2f}% · "
        "TTFT P95 "
        f"+{comparison['light_vs_vllm_ttft_p95_delta_ms']:.2f} ms · "
        "最大 ITL P95 "
        f"+{comparison['light_vs_vllm_max_itl_p95_delta_ms']:.2f} ms",
        ha="center",
        fontsize=10.5,
        color="#334155",
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.035,
        "Qwen2.5-Coder-7B-Instruct · BF16 · TP=2 · 2×RTX 5090（SYS，无 P2P） · 固定 16×512 decode\n"
        "light/vLLM 交替重启各 3 次；柱为正式轮次中位数，圆点为逐轮结果，warmup 不计。",
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
    render(json.loads(args.summary.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
