"""把稳定性能快照编码成 Prometheus 文本格式。"""

from __future__ import annotations

from light_vllm.runtime.observability.interfaces import (
    HistogramSnapshot,
    PerformanceSnapshot,
)

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _escaped_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _labels(**values: str) -> str:
    encoded = ",".join(f'{name}="{_escaped_label(value)}"' for name, value in values.items())
    return f"{{{encoded}}}" if encoded else ""


def _metric(
    lines: list[str],
    name: str,
    kind: str,
    help_text: str,
    value: int | float,
    *,
    labels: dict[str, str],
) -> None:
    lines.extend(
        (
            f"# HELP {name} {help_text}",
            f"# TYPE {name} {kind}",
            f"{name}{_labels(**labels)} {value}",
        )
    )


def _histogram(
    lines: list[str],
    name: str,
    help_text: str,
    snapshot: HistogramSnapshot,
    *,
    labels: dict[str, str],
) -> None:
    lines.extend((f"# HELP {name} {help_text}", f"# TYPE {name} histogram"))
    for bound, count in zip(
        snapshot.bounds,
        snapshot.cumulative_counts,
        strict=True,
    ):
        lines.append(f"{name}_bucket{_labels(**labels, le=str(bound))} {count}")
    lines.extend(
        (
            f"{name}_bucket{_labels(**labels, le='+Inf')} {snapshot.count}",
            f"{name}_count{_labels(**labels)} {snapshot.count}",
            f"{name}_sum{_labels(**labels)} {snapshot.total}",
        )
    )


def render_prometheus(snapshot: PerformanceSnapshot) -> str:
    """只做表达转换；采集时不触碰 Engine、Scheduler 或 KV 内部状态。"""

    lines: list[str] = []
    labels = {"model": snapshot.model_name}
    scheduler = snapshot.scheduler
    _metric(
        lines,
        "light_vllm_requests_waiting",
        "gauge",
        "Requests waiting for a scheduler slot.",
        scheduler.waiting_requests,
        labels=labels,
    )
    _metric(
        lines,
        "light_vllm_requests_running",
        "gauge",
        "Requests admitted to the scheduler running set.",
        scheduler.running_requests,
        labels=labels,
    )
    _metric(
        lines,
        "light_vllm_queue_tokens",
        "gauge",
        "Conservative remaining token budget of waiting requests.",
        scheduler.waiting_token_budget,
        labels=labels,
    )
    _metric(
        lines,
        "light_vllm_running_tokens",
        "gauge",
        "Conservative remaining token budget of running requests.",
        scheduler.running_token_budget,
        labels=labels,
    )

    cache = scheduler.kv_cache
    if cache.usage_ratio is not None:
        _metric(
            lines,
            "light_vllm_kv_cache_usage_ratio",
            "gauge",
            "Non-reclaimable KV token slots divided by total token slots.",
            cache.usage_ratio,
            labels=labels,
        )
        _metric(
            lines,
            "light_vllm_kv_cache_used_token_slots",
            "gauge",
            "KV token slots that cannot be reclaimed immediately.",
            cache.used_token_slots,
            labels=labels,
        )
        _metric(
            lines,
            "light_vllm_kv_cache_capacity_token_slots",
            "gauge",
            "Total KV cache capacity in token slots.",
            cache.capacity_token_slots,
            labels=labels,
        )

    _histogram(
        lines,
        "light_vllm_time_to_first_token_seconds",
        "Time from request admission to the first visible token.",
        snapshot.time_to_first_token,
        labels=labels,
    )
    _histogram(
        lines,
        "light_vllm_time_per_output_token_seconds",
        "Time between visible output tokens after the first token.",
        snapshot.time_per_output_token,
        labels=labels,
    )
    _metric(
        lines,
        "light_vllm_prompt_tokens_total",
        "counter",
        "Prompt tokens of requests that emitted at least one token.",
        snapshot.prompt_tokens_total,
        labels=labels,
    )
    _metric(
        lines,
        "light_vllm_generation_tokens_total",
        "counter",
        "Visible output tokens emitted by the engine.",
        snapshot.generation_tokens_total,
        labels=labels,
    )
    lines.extend(
        (
            "# HELP light_vllm_requests_total Requests by terminal outcome.",
            "# TYPE light_vllm_requests_total counter",
        )
    )
    for outcome, value in (
        ("finished", snapshot.finished_requests_total),
        ("failed", snapshot.failed_requests_total),
        ("cancelled", snapshot.cancelled_requests_total),
    ):
        lines.append(f"light_vllm_requests_total{_labels(**labels, outcome=outcome)} {value}")
    return "\n".join(lines) + "\n"
