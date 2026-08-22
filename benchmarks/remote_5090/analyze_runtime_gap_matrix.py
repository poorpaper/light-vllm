from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COUNTERS = {
    "light_pauses": "light_vllm_nonpreemptive_pauses_total",
    "light_resubmits": "light_vllm_self_resubmits_total",
    "light_rolled_back_tokens": "light_vllm_self_resubmit_rolled_back_tokens_total",
    "vllm_preemptions": "vllm:num_preemptions_total",
}
SCALARS = (
    "output_tokens_per_s",
    "ttft_p50_s",
    "ttft_p95_s",
    "tpot_p50_s",
    "tpot_p95_s",
    "max_itl_p50_s",
    "max_itl_p95_s",
    "max_itl_max_s",
    "e2e_p50_s",
    "e2e_p95_s",
)


@dataclass(frozen=True)
class Configuration:
    key: str
    label: str
    directory: Path
    mode: str


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prometheus_value(path: Path, metric: str) -> float | None:
    if not path.exists():
        return None
    match = re.search(
        rf"^{re.escape(metric)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)$",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return float(match.group(1)) if match is not None else None


def _counter_delta(directory: Path, prefix: str, metric: str) -> float:
    before = _prometheus_value(directory / f"{prefix}.metrics-before.prom", metric)
    after = _prometheus_value(directory / f"{prefix}.metrics-after.prom", metric)
    if before is None or after is None:
        return 0.0
    return after - before


def _run(configuration: Configuration, case: str, run: int) -> dict[str, Any]:
    prefix = f"{configuration.mode}-{case}-r{run}"
    payload = _load(configuration.directory / f"{prefix}.json")
    summary = payload["summary"]
    return {
        "run": run,
        "requests": int(summary["requests"]),
        "successful_requests": int(summary["successful_requests"]),
        "failed_requests": int(summary["failed_requests"]),
        "output_tokens": int(summary["output_tokens"]),
        **{key: float(summary[key]) if summary[key] is not None else None for key in SCALARS},
        "counters": {
            key: _counter_delta(configuration.directory, prefix, metric)
            for key, metric in COUNTERS.items()
        },
    }


def _median(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(statistics.median(present)) if present else None


def _median_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "requests": int(statistics.median(run["requests"] for run in runs)),
        "successful_requests": int(statistics.median(run["successful_requests"] for run in runs)),
        "failed_requests": int(statistics.median(run["failed_requests"] for run in runs)),
        "output_tokens": int(statistics.median(run["output_tokens"] for run in runs)),
        **{key: _median([run[key] for run in runs]) for key in SCALARS},
        "counters": {
            key: float(statistics.median(run["counters"][key] for run in runs)) for key in COUNTERS
        },
    }


def _available(configuration: Configuration, case: str, runs: int) -> bool:
    return all(
        (configuration.directory / f"{configuration.mode}-{case}-r{run}.json").exists()
        for run in range(1, runs + 1)
    )


def _profile_summary(configuration: Configuration, case: str) -> dict[str, Any] | None:
    path = configuration.directory / f"{configuration.mode}-{case}-profile-stages.json"
    if not path.exists():
        return None
    payload = _load(path)
    steps = payload.get("steps", [])
    groups: dict[str, list[dict[str, Any]]] = {
        "decode_width_1": [],
        "mixed_padded": [],
        "prefill_width_2_64": [],
        "prefill_width_65_256": [],
        "prefill_width_257_plus": [],
    }
    for step in steps:
        width = step.get("query_width")
        batch = step.get("batch_size")
        tokens = step.get("model_tokens")
        if width is None or batch is None or tokens is None:
            continue
        if width == 1:
            group = "decode_width_1"
        elif tokens != batch * width:
            group = "mixed_padded"
        elif width <= 64:
            group = "prefill_width_2_64"
        elif width <= 256:
            group = "prefill_width_65_256"
        else:
            group = "prefill_width_257_plus"
        groups[group].append(step)

    grouped = {}
    for name, values in groups.items():
        if not values:
            continue
        grouped[name] = {
            "steps": len(values),
            "batch_size_p50": _median([float(value["batch_size"]) for value in values]),
            "query_width_p50": _median([float(value["query_width"]) for value in values]),
            "executor_wall_p50_ms": _median([value["executor_wall_ms"] for value in values]),
            "cuda_event_p50_ms": _median([value["cuda_event_ms"] for value in values]),
            "previous_gap_p50_ms": _median([value["previous_executor_gap_ms"] for value in values]),
        }
    stages = payload.get("stages", {})
    executor_steps = int(payload.get("executor_steps", 0))

    def per_step(stage: str) -> float | None:
        value = stages.get(stage, {}).get("wall_total_ms")
        return float(value) / executor_steps if value is not None and executor_steps else None

    return {
        "path": str(path),
        "executor_steps": executor_steps,
        "executor_wall_per_step_ms": (
            float(payload["executor_wall_total_ms"]) / executor_steps if executor_steps else None
        ),
        "cuda_event_per_step_ms": (
            float(payload["executor_cuda_event_total_ms"]) / executor_steps
            if executor_steps
            else None
        ),
        "inter_executor_gap_per_step_ms": (
            float(payload["inter_executor_gap_total_ms"]) / max(executor_steps - 1, 1)
            if executor_steps
            else None
        ),
        "stage_wall_per_executor_step_ms": {
            stage: per_step(stage)
            for stage in (
                "scheduler.schedule",
                "engine.build_batch",
                "engine.executor_future_total",
                "engine.validate_output",
                "engine.apply_output",
                "paged_step.forward",
                "worker.model_forward",
                "sampler.sample",
                "paged_metadata.validate",
                "paged_metadata.visibility_tensor",
            )
            if stage in stages
        },
        "shape_groups": grouped,
    }


def _bar_panel(axis, records, metric: str, title: str, *, scale: float = 1.0) -> None:
    labels = [record["label"] for record in records]
    values = [record["median"][metric] * scale for record in records]
    colors = ["#7A7A7A", "#2F6B9A", "#D88932", "#A84448"][: len(records)]
    bars = axis.bar(range(len(records)), values, color=colors)
    axis.set_title(title)
    axis.set_xticks(range(len(records)), labels, rotation=18, ha="right")
    axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)


def _plot_runtime(summary: dict[str, Any], output: Path) -> None:
    cases = summary["cases"]
    fixed = cases.get("fixed", [])
    low = cases.get("normal-low", [])
    loaded = cases.get("normal-loaded", [])
    if not fixed or not low or not loaded:
        return
    import matplotlib.pyplot as plt

    for record in fixed:
        record["median"]["effective_step_ms"] = (
            16.0 / record["median"]["output_tokens_per_s"] * 1000.0
        )
    figure, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    _bar_panel(axes[0, 0], fixed, "effective_step_ms", "Fixed B16/W1 service step (ms)")
    _bar_panel(axes[0, 1], low, "ttft_p95_s", "Normal 2 rps TTFT p95 (ms)", scale=1000)
    _bar_panel(axes[0, 2], low, "max_itl_p95_s", "Normal 2 rps max ITL p95 (ms)", scale=1000)
    _bar_panel(axes[1, 0], loaded, "output_tokens_per_s", "Loaded output throughput (tok/s)")
    _bar_panel(axes[1, 1], loaded, "ttft_p95_s", "Loaded TTFT p95 (ms)", scale=1000)
    _bar_panel(axes[1, 2], loaded, "tpot_p95_s", "Loaded TPOT p95 (ms)", scale=1000)
    figure.suptitle("Qwen2.5-Coder-7B on RTX 5090: runtime gap with policy effects disabled")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_policy(summary: dict[str, Any], output: Path) -> None:
    records = summary["cases"].get("pressure", [])
    if not records:
        return
    import matplotlib.pyplot as plt

    for record in records:
        median = record["median"]
        median["success_percent"] = 100.0 * median["successful_requests"] / median["requests"]
        median["rollback_events"] = (
            median["counters"]["light_resubmits"] + median["counters"]["vllm_preemptions"]
        )
    figure, axes = plt.subplots(1, 5, figsize=(23, 5.2))
    _bar_panel(axes[0], records, "success_percent", "Success rate (%)")
    _bar_panel(axes[1], records, "output_tokens_per_s", "Output throughput (tok/s)")
    _bar_panel(axes[2], records, "ttft_p95_s", "TTFT p95 (ms)", scale=1000)
    _bar_panel(axes[3], records, "max_itl_p95_s", "Max ITL p95 (ms)", scale=1000)
    _bar_panel(axes[4], records, "rollback_events", "Rollback / preemption events")
    figure.suptitle("4096-token KV pressure under Poisson arrivals: policy trade-off")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    runtime_configurations = (
        Configuration("baseline", "Light baseline", args.baseline, "light-strict"),
        Configuration("candidate", "Light fixed", args.candidate, "light-strict"),
        Configuration("vllm_eager", "vLLM eager", args.vllm, "vllm-eager"),
        Configuration("vllm_default", "vLLM graph", args.vllm, "vllm-default"),
    )
    pressure_configurations = (
        Configuration("candidate_strict", "Light strict", args.candidate, "light-strict"),
        Configuration(
            "candidate_optimistic",
            "Light self-resubmit",
            args.candidate,
            "light-optimistic",
        ),
        Configuration("vllm_default", "vLLM preempt16", args.vllm, "vllm-default"),
        Configuration("vllm_cap4", "vLLM cap4", args.vllm, "vllm-nopreempt4"),
    )
    summary: dict[str, Any] = {"runs": args.runs, "cases": {}, "profiles": {}}
    for case in ("fixed", "normal-low", "normal-loaded", "pressure"):
        configurations = pressure_configurations if case == "pressure" else runtime_configurations
        records = []
        for configuration in configurations:
            if not _available(configuration, case, args.runs):
                continue
            runs = [_run(configuration, case, run) for run in range(1, args.runs + 1)]
            records.append(
                {
                    "key": configuration.key,
                    "label": configuration.label,
                    "mode": configuration.mode,
                    "runs": runs,
                    "median": _median_runs(runs),
                }
            )
            profile = _profile_summary(configuration, case)
            if profile is not None:
                summary["profiles"][f"{configuration.key}:{case}"] = profile
        summary["cases"][case] = records

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_runtime(summary, args.output_dir / "runtime_gap.png")
    _plot_policy(summary, args.output_dir / "policy_tradeoff.png")


if __name__ == "__main__":
    main()
