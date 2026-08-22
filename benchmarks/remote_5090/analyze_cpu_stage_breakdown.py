from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_total(payload: dict[str, Any], name: str, output_steps: float) -> float:
    return float(payload["stages"][name]["wall_total_ms"]) / output_steps


def _service_step_ms(payload: dict[str, Any], batch_size: int) -> float:
    return batch_size / float(payload["summary"]["output_tokens_per_s"]) * 1000


def _plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    light = summary["light-vllm"]
    vllm = summary["vLLM eager"]
    breakdown = light["inter_executor_breakdown_ms"]
    categories = list(breakdown)
    colors = plt.get_cmap("tab20").colors
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.3), constrained_layout=True)

    bottom = 0.0
    for index, category in enumerate(categories):
        value = breakdown[category]
        axes[0].bar("light-vLLM", value, bottom=bottom, label=category, color=colors[index])
        bottom += value
    axes[0].set_title("light inter-executor gap")
    axes[0].set_ylabel("Time per 16-token output step (ms)")
    axes[0].text(0, bottom, f"{bottom:.3f}", ha="center", va="bottom")
    axes[0].legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axes[0].grid(axis="y", alpha=0.25)

    gap_values = [light["inter_executor_gap_ms"], vllm["inter_step_gap_ms"]]
    axes[1].bar(("light-vLLM", "vLLM eager"), gap_values, color=("#E45756", "#4C78A8"))
    axes[1].set_title("Gap before the next device step")
    axes[1].set_ylabel("Milliseconds")
    axes[1].grid(axis="y", alpha=0.25)
    for index, value in enumerate(gap_values):
        axes[1].text(index, value, f"{value:.3f}", ha="center", va="bottom")

    build = light["build_batch_quartile_ms"]
    validation = light["request_validation_quartile_ms"]
    positions = range(1, 5)
    axes[2].plot(positions, build, marker="o", label="Build ExecutionBatch")
    axes[2].plot(positions, validation, marker="o", label="ExecutionRequest validation")
    axes[2].set_xticks(tuple(positions), ("Q1", "Q2", "Q3", "Q4"))
    axes[2].set_title("Cost grows with generated context")
    axes[2].set_ylabel("Milliseconds per step")
    axes[2].set_xlabel("Chronological decode quartile")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    figure.suptitle("Steady decode CPU/runtime breakdown on RTX 5090", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--light-lean", type=Path, required=True)
    parser.add_argument("--light-full", type=Path, required=True)
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--light-service", type=Path, required=True)
    parser.add_argument("--vllm-service", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    light_lean = _load(args.light_lean)
    light_full = _load(args.light_full)
    vllm = _load(args.vllm)
    light_service = _load(args.light_service)
    vllm_service = _load(args.vllm_service)
    output_steps = float(light_service["summary"]["output_tokens"]) / args.batch_size

    light_gap_ms = float(light_lean["inter_executor_gap_total_ms"]) / output_steps
    vllm_gap_ms = float(vllm["inter_step_gap_total_ms"]) / output_steps
    publish_mean_ms = float(light_lean["stages"]["engine.publish_scheduler_stats"]["wall_mean_ms"])
    breakdown = {
        "Build ExecutionBatch": _stage_total(light_lean, "engine.build_batch", output_steps),
        "asyncio.to_thread handoff": (
            float(light_lean["stages"]["engine.to_thread_total"]["wall_total_ms"])
            - float(light_lean["executor_wall_total_ms"])
        )
        / output_steps,
        "Apply output": _stage_total(light_lean, "engine.apply_output", output_steps),
        "Scheduler": _stage_total(light_lean, "scheduler.schedule", output_steps),
        "HTTP SSE encode": float(
            light_full["interval_overlap"]["http.encode_event"]["outside_executor_ms"]
        )
        / output_steps,
        "Finish execution": _stage_total(light_lean, "engine.finish_execution", output_steps),
        "Validate output": _stage_total(light_lean, "engine.validate_output", output_steps),
        "Step observer": _stage_total(light_lean, "observer.step_completed", output_steps),
        "Acquire/release": (
            float(light_lean["stages"]["executor.acquire"]["wall_total_ms"])
            + float(light_lean["stages"]["executor.lease_release"]["wall_total_ms"])
        )
        / output_steps,
        "Standalone stats snapshot": publish_mean_ms,
    }
    measured_ms = sum(breakdown.values())
    breakdown["Event loop/locks/unclassified"] = max(light_gap_ms - measured_ms, 0.0)

    request_quartiles = [
        float(value) * args.batch_size
        for value in light_full["stages"]["execution_request.validate"]["wall_quartile_means_ms"]
    ]
    build_quartiles = [
        float(value)
        for value in light_full["stages"]["engine.build_batch"]["wall_quartile_means_ms"]
    ]
    light_service_ms = _service_step_ms(light_service, args.batch_size)
    vllm_service_ms = _service_step_ms(vllm_service, args.batch_size)
    summary = {
        "scope": {
            "model": "Qwen2.5-Coder-7B-Instruct",
            "device": "RTX 5090",
            "batch_size": args.batch_size,
            "prompt_tokens_per_request": 16,
            "output_tokens_per_request": 512,
            "output_steps": output_steps,
            "vllm_mode": "eager",
        },
        "notes": [
            "The lean light instrumentation had no measurable slowdown versus its inactive run.",
            "The full light pass adds overhead; only nested body durations and overlap are used.",
            "Wall stages are nested where documented and must not all be added together.",
            "vLLM uses step_with_batch_queue, so model execution, waiting, and sampling overlap.",
        ],
        "light-vllm": {
            "service_step_ms": light_service_ms,
            "executor_span_ms": float(light_lean["executor_span_ms"]) / output_steps,
            "executor_wall_ms": float(light_lean["executor_wall_total_ms"]) / output_steps,
            "executor_cuda_event_ms": float(light_lean["executor_cuda_event_total_ms"])
            / output_steps,
            "inter_executor_gap_ms": light_gap_ms,
            "inter_executor_breakdown_ms": breakdown,
            "build_batch_quartile_ms": build_quartiles,
            "request_validation_quartile_ms": request_quartiles,
            "request_validation_mean_ms": _stage_total(
                light_full, "execution_request.validate", output_steps
            ),
            "apply_output_nested_ms": {
                "scheduler.complete": _stage_total(light_full, "scheduler.complete", output_steps),
                "tokens_generated observer": _stage_total(
                    light_full, "observer.tokens_generated", output_steps
                ),
                "scheduler stats snapshot": publish_mean_ms,
            },
        },
        "vLLM eager": {
            "service_step_ms": vllm_service_ms,
            "engine_step_span_ms": float(vllm["engine_step_span_ms"]) / output_steps,
            "inter_step_gap_ms": vllm_gap_ms,
            "stages_ms": {
                "scheduler.schedule": _stage_total(vllm, "scheduler.schedule", output_steps),
                "executor.execute_model": _stage_total(
                    vllm, "executor.execute_model", output_steps
                ),
                "executor.future_result": _stage_total(
                    vllm, "executor.future_result", output_steps
                ),
                "executor.sample_tokens": _stage_total(
                    vllm, "executor.sample_tokens", output_steps
                ),
                "scheduler.update_from_output": _stage_total(
                    vllm, "scheduler.update_from_output", output_steps
                ),
            },
        },
        "derived": {
            "service_gap_light_minus_vllm_ms": light_service_ms - vllm_service_ms,
            "inter_step_gap_light_minus_vllm_ms": light_gap_ms - vllm_gap_ms,
            "inter_step_gap_ratio_light_over_vllm": light_gap_ms / vllm_gap_ms,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "cpu_stage_breakdown.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(summary, args.output_dir / "figures" / "09_cpu_stage_breakdown.png")


if __name__ == "__main__":
    main()
