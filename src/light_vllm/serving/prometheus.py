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
    include_metadata: bool = True,
) -> None:
    if include_metadata:
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
        "light_vllm_short_launch_requests",
        "gauge",
        "Admitted short requests still waiting for their first visible token.",
        scheduler.short_launch_requests,
        labels=labels,
    )
    _metric(
        lines,
        "light_vllm_waiting_pending_tokens",
        "gauge",
        "Known input tokens not yet computed for waiting requests.",
        scheduler.waiting_pending_tokens,
        labels=labels,
    )
    _metric(
        lines,
        "light_vllm_running_pending_tokens",
        "gauge",
        "Known input tokens not yet computed for running requests.",
        scheduler.running_pending_tokens,
        labels=labels,
    )
    _metric(
        lines,
        "light_vllm_waiting_max_remaining_tokens",
        "gauge",
        "Conservative remaining token upper bound of waiting requests.",
        scheduler.waiting_max_remaining_tokens,
        labels=labels,
    )
    _metric(
        lines,
        "light_vllm_running_max_remaining_tokens",
        "gauge",
        "Conservative remaining token upper bound of running requests.",
        scheduler.running_max_remaining_tokens,
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
            "light_vllm_kv_cache_claimed_token_slots",
            "gauge",
            "KV token slots promised to admitted requests but not allocated yet.",
            cache.claimed_token_slots,
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

    _metric(
        lines,
        "light_vllm_self_resubmits_total",
        "counter",
        "Requests that released their own KV and re-entered scheduling.",
        scheduler.self_resubmits_total,
        labels=labels,
    )
    _metric(
        lines,
        "light_vllm_self_resubmit_rolled_back_tokens_total",
        "counter",
        "Computed-token progress rolled back by self-resubmit.",
        scheduler.self_resubmit_rolled_back_tokens_total,
        labels=labels,
    )
    for name, help_text, value in (
        (
            "light_vllm_speculation_attempts_total",
            "Completed speculative verification attempts.",
            snapshot.speculation_attempts_total,
        ),
        (
            "light_vllm_speculation_hits_total",
            "Speculative attempts that accepted at least one draft node.",
            snapshot.speculation_hits_total,
        ),
        (
            "light_vllm_speculative_proposed_nodes_total",
            "Draft tree nodes sent to the target model.",
            snapshot.speculative_proposed_nodes_total,
        ),
        (
            "light_vllm_speculative_accepted_nodes_total",
            "Draft tree nodes accepted by the target model.",
            snapshot.speculative_accepted_nodes_total,
        ),
        (
            "light_vllm_speculative_verified_tokens_total",
            "Tokens produced by speculative target verification, including the final target token.",
            snapshot.speculative_verified_tokens_total,
        ),
        (
            "light_vllm_speculative_draft_roots_total",
            "Root nodes across proposed draft trees.",
            snapshot.speculative_draft_roots_total,
        ),
        (
            "light_vllm_speculative_branching_parents_total",
            "Draft nodes that had more than one proposed child.",
            snapshot.speculative_branching_parents_total,
        ),
        (
            "light_vllm_speculative_compacted_tokens_total",
            "Accepted draft tokens moved while compacting physical KV.",
            snapshot.speculative_compacted_tokens_total,
        ),
    ):
        _metric(lines, name, "counter", help_text, value, labels=labels)
    _metric(
        lines,
        "light_vllm_speculative_max_draft_depth",
        "gauge",
        "Maximum observed draft tree depth.",
        snapshot.speculative_max_draft_depth,
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
        "light_vllm_inter_token_latency_seconds",
        "Visible interval between output tokens after the first token.",
        snapshot.inter_token_latency,
        labels=labels,
    )
    for index, step in enumerate(snapshot.step_latency):
        token_bucket = (
            str(step.max_model_tokens_computed)
            if step.max_model_tokens_computed is not None
            else "+Inf"
        )
        _histogram(
            lines,
            "light_vllm_engine_step_seconds",
            "Completed device-step latency grouped by actual model-token upper bound.",
            step.latency,
            labels={**labels, "model_tokens_computed_le": token_bucket},
            include_metadata=index == 0,
        )
    _metric(
        lines,
        "light_vllm_prompt_tokens_total",
        "counter",
        "Prompt tokens admitted to the engine.",
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
        ("rejected", snapshot.rejected_requests_total),
        ("overloaded", snapshot.overloaded_requests_total),
    ):
        lines.append(f"light_vllm_requests_total{_labels(**labels, outcome=outcome)} {value}")
    return "\n".join(lines) + "\n"
