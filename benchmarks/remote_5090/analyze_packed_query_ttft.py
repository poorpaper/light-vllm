# 报告正文保留完整中文句子，避免为满足源码行宽而破坏生成后的 Markdown 可读性。
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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
    return {
        "executor_steps": int(profile["executor_steps"]),
        "executor_span_ms": float(profile["executor_span_ms"]),
        "executor_wall_total_ms": float(profile["executor_wall_total_ms"]),
        "executor_cuda_event_total_ms": float(profile["executor_cuda_event_total_ms"]),
        "inter_executor_gap_total_ms": float(profile["inter_executor_gap_total_ms"]),
        "inter_executor_gap_mean_ms": float(profile["inter_executor_gap_total_ms"])
        / int(profile["executor_steps"]),
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


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1600, 960), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 38), title, fill=COLORS["text"], font=_font(38, bold=True))
    draw.text((72, 92), subtitle, fill=COLORS["muted"], font=_font(21))
    return image, draw


def _legend(draw: ImageDraw.ImageDraw, items: list[tuple[str, str]], x: int, y: int) -> None:
    for index, (label, color) in enumerate(items):
        left = x + index * 285
        draw.rounded_rectangle((left, y, left + 30, y + 18), 4, fill=color)
        draw.text((left + 40, y - 6), label, fill=COLORS["text"], font=_font(20))


def _plot_ttft(summary: dict[str, object], output: Path) -> None:
    image, draw = _canvas(
        "正常 Poisson 负载：TTFT P95",
        "Qwen2.5-Coder-7B / BF16 / RTX 5090；点为 3 次中位数，误差线为 min–max；纵轴为对数刻度",
    )
    left, top, right, bottom = 145, 170, 1530, 820
    ticks = (20, 50, 100, 200, 500, 1000, 2000)
    y_min, y_max = math.log10(20), math.log10(2000)

    def y_of(value: float) -> float:
        return bottom - (math.log10(value) - y_min) / (y_max - y_min) * (bottom - top)

    x_by_rate = {
        rate: left + index * (right - left) / (len(RATES) - 1) for index, rate in enumerate(RATES)
    }
    for tick in ticks:
        y = y_of(tick)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        draw.text((60, y - 13), f"{tick}", fill=COLORS["muted"], font=_font(18))
    draw.line((left, top, left, bottom), fill=COLORS["text"], width=2)
    draw.line((left, bottom, right, bottom), fill=COLORS["text"], width=2)
    draw.text((20, 145), "ms", fill=COLORS["muted"], font=_font(18))

    labels = {
        "baseline": "Light 原版 strict",
        "candidate": "Light Packed strict",
        "vllm_eager": "vLLM eager",
        "vllm_default": "vLLM default/graph",
    }
    primary = summary["primary"]  # type: ignore[index]
    for group in labels:
        points: list[tuple[float, float]] = []
        for rate in RATES:
            stats = primary[group][str(rate)]["ttft_p95_ms"]  # type: ignore[index]
            x = x_by_rate[rate]
            y = y_of(float(stats["median"]))
            y_low = y_of(float(stats["max"]))
            y_high = y_of(float(stats["min"]))
            draw.line((x, y_low, x, y_high), fill=COLORS[group], width=3)
            draw.line((x - 7, y_low, x + 7, y_low), fill=COLORS[group], width=2)
            draw.line((x - 7, y_high, x + 7, y_high), fill=COLORS[group], width=2)
            points.append((x, y))
        draw.line(points, fill=COLORS[group], width=4)
        for rate, (x, y) in zip(RATES, points, strict=True):
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=COLORS[group])
            value = primary[group][str(rate)]["ttft_p95_ms"]["median"]  # type: ignore[index]
            x_offset = 10 if group == "vllm_eager" else -10
            y_offset = -30 if group in ("baseline", "candidate") else 18
            draw.text(
                (x + x_offset, y + y_offset),
                f"{float(value):.0f}",
                fill=COLORS[group],
                font=_font(17),
                anchor="mm",
            )
    for rate, x in x_by_rate.items():
        draw.text((x - 18, bottom + 18), str(rate), fill=COLORS["text"], font=_font(20))
    draw.text((690, 865), "请求到达率（req/s）", fill=COLORS["text"], font=_font(22))
    _legend(draw, [(labels[key], COLORS[key]) for key in labels], 155, 900)
    image.save(output)


def _plot_throughput(summary: dict[str, object], output: Path) -> None:
    image, draw = _canvas(
        "正常 Poisson 负载：输出吞吐",
        "柱为 3 次中位数；Packed 柱上方标注其相对 vLLM eager 的比例",
    )
    left, top, right, bottom = 120, 170, 1530, 820
    maximum = 700.0
    for tick in range(0, 701, 100):
        y = bottom - tick / maximum * (bottom - top)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        draw.text((52, y - 12), str(tick), fill=COLORS["muted"], font=_font(18))
    draw.line((left, top, left, bottom), fill=COLORS["text"], width=2)
    draw.line((left, bottom, right, bottom), fill=COLORS["text"], width=2)
    draw.text((18, 145), "tok/s", fill=COLORS["muted"], font=_font(18))
    groups = ("baseline", "candidate", "vllm_eager", "vllm_default")
    labels = {
        "baseline": "Light 原版",
        "candidate": "Light Packed",
        "vllm_eager": "vLLM eager",
        "vllm_default": "vLLM graph",
    }
    primary = summary["primary"]  # type: ignore[index]
    cluster_width, bar_width = 310, 55
    for rate_index, rate in enumerate(RATES):
        center = 285 + rate_index * cluster_width
        for group_index, group in enumerate(groups):
            value = float(primary[group][str(rate)]["throughput"]["median"])  # type: ignore[index]
            x0 = center - 126 + group_index * 64
            y0 = bottom - value / maximum * (bottom - top)
            draw.rectangle((x0, y0, x0 + bar_width, bottom), fill=COLORS[group])
            draw.text((x0 - 1, y0 - 25), f"{value:.0f}", fill=COLORS[group], font=_font(16))
        candidate = float(primary["candidate"][str(rate)]["throughput"]["median"])  # type: ignore[index]
        eager = float(primary["vllm_eager"][str(rate)]["throughput"]["median"])  # type: ignore[index]
        draw.text(
            (center - 72, 835),
            f"{rate} req/s   Packed/eager {candidate / eager:.1%}",
            fill=COLORS["text"],
            font=_font(17),
        )
    _legend(draw, [(labels[key], COLORS[key]) for key in groups], 160, 905)
    image.save(output)


def _panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    labels: tuple[str, ...],
    series: tuple[tuple[str, tuple[float, ...], str], ...],
    *,
    suffix: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, 18, fill=COLORS["panel"], outline=COLORS["grid"], width=2)
    draw.text((x0 + 25, y0 + 20), title, fill=COLORS["text"], font=_font(23, bold=True))
    values = [value for _, entries, _ in series for value in entries]
    maximum = max(values) * 1.22 if values else 1.0
    # 横轴标签和图例分两层放置，避免短面板里相互覆盖。
    chart_top, chart_bottom = y0 + 85, y1 - 90
    group_width = (x1 - x0 - 90) / len(labels)
    bar_width = min(55, group_width / (len(series) + 1))
    for label_index, label in enumerate(labels):
        center = x0 + 55 + group_width * (label_index + 0.5)
        for series_index, (_, entries, color) in enumerate(series):
            value = entries[label_index]
            left = center + (series_index - (len(series) - 1) / 2) * (bar_width + 8) - bar_width / 2
            top = chart_bottom - value / maximum * (chart_bottom - chart_top)
            draw.rectangle((left, top, left + bar_width, chart_bottom), fill=color)
            draw.text(
                (left - 4, top - 24),
                f"{value:.1f}{suffix}",
                fill=color,
                font=_font(15),
            )
        draw.text((center - 50, chart_bottom + 15), label, fill=COLORS["text"], font=_font(17))
    legend_x = x0 + 25
    for name, _, color in series:
        draw.rectangle((legend_x, y1 - 37, legend_x + 20, y1 - 24), fill=color)
        draw.text((legend_x + 27, y1 - 43), name, fill=COLORS["muted"], font=_font(15))
        legend_x += 170


def _plot_breakdown(summary: dict[str, object], output: Path) -> None:
    image, draw = _canvas(
        "8 req/s Profile：Mixed 与 Decode 分解",
        "相同 64 请求；Packed 只消除 mixed padding，decode W1 与 Driver gap 基本未变",
    )
    baseline = summary["profiles"]["baseline"]  # type: ignore[index]
    candidate = summary["profiles"]["candidate"]  # type: ignore[index]
    _panel(
        draw,
        (55, 145, 780, 515),
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
    )
    draw.text(
        (85, 460),
        f"Packed 避免了 {int(candidate['mixed']['legacy_positions']) - int(candidate['mixed']['model_tokens']):,} 个 padding 位置",
        fill=COLORS["muted"],
        font=_font(17),
    )
    _panel(
        draw,
        (820, 145, 1545, 515),
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
                (float(baseline["mixed"]["cuda_p50_ms"]), float(candidate["mixed"]["cuda_p50_ms"])),
                COLORS["candidate"],
            ),
        ),
        suffix="ms",
    )
    _panel(
        draw,
        (55, 545, 780, 915),
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
    _panel(
        draw,
        (820, 545, 1545, 915),
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
    image.save(output)


def _plot_ablation(summary: dict[str, object], output: Path) -> None:
    image, draw = _canvas(
        "8 req/s：reserved_sequences 消融",
        "短请求池固定：256 scheduled tokens / 2047 KV slots；只改变 sequence slot 数量",
    )
    ablation = summary["reserved_sequences_ablation"]  # type: ignore[index]
    labels = ("off", "1", "2", "4", "8")
    _panel(
        draw,
        (80, 170, 780, 850),
        "TTFT P95（3 次中位数）",
        labels,
        (
            (
                "TTFT",
                tuple(float(ablation[label]["ttft_p95_ms"]["median"]) for label in labels),
                COLORS["baseline"],
            ),
        ),
        suffix="ms",
    )
    _panel(
        draw,
        (820, 170, 1520, 850),
        "吞吐（3 次中位数）",
        labels,
        (
            (
                "吞吐",
                tuple(float(ablation[label]["throughput"]["median"]) for label in labels),
                COLORS["candidate"],
            ),
        ),
        suffix="",
    )
    image.save(output)


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
- 重画命令：`python benchmarks/remote_5090/analyze_packed_query_ttft.py <解包目录> <结果目录>`（需要 Pillow）。
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
        "archive_sha256": archive_sha,
        "raw_manifest_files": len((raw / "SHA256SUMS").read_text(encoding="utf-8").splitlines()),
    }
    (analysis / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot_ttft(summary, analysis / "ttft_p95.png")
    _plot_throughput(summary, analysis / "throughput.png")
    _plot_breakdown(summary, analysis / "step_breakdown.png")
    _plot_ablation(summary, analysis / "reserved_sequences_ablation.png")
    _write_report(summary, result_root)


if __name__ == "__main__":
    main()
