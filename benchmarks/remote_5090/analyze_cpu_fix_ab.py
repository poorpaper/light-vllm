from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _first_mismatch_positions(left: dict[str, Any], right: dict[str, Any]) -> list[int]:
    right_by_id = {result["request_id"]: result for result in right["results"]}
    positions: list[int] = []
    for left_result in left["results"]:
        right_result = right_by_id[left_result["request_id"]]
        left_tokens = left_result["generated_token_ids"]
        right_tokens = right_result["generated_token_ids"]
        mismatch = next(
            (
                index
                for index, (left_token, right_token) in enumerate(
                    zip(left_tokens, right_tokens, strict=True)
                )
                if left_token != right_token
            ),
            -1,
        )
        positions.append(mismatch)
    return positions


def _lean_run(input_dir: Path, side: str, run: int, *, output_steps: int) -> dict[str, Any]:
    result = _load(input_dir / f"{side}-lean-r{run}.json")
    profile = _load(input_dir / f"{side}-lean-r{run}-stages.json")
    throughput = float(result["summary"]["output_tokens_per_s"])
    build = profile["stages"]["engine.build_batch"]
    return {
        "run": run,
        "successful_requests": int(result["summary"]["successful_requests"]),
        "output_tokens": int(result["summary"]["output_tokens"]),
        "throughput_tokens_per_s": throughput,
        "service_step_ms": 16_000.0 / throughput,
        "build_batch_ms": float(build["wall_total_ms"]) / output_steps,
        "build_batch_quartile_ms": [float(value) for value in build["wall_quartile_means_ms"]],
        "inter_executor_gap_ms": float(profile["inter_executor_gap_total_ms"]) / output_steps,
        "executor_wall_ms": float(profile["executor_wall_total_ms"]) / output_steps,
        "executor_cuda_event_ms": float(profile["executor_cuda_event_total_ms"]) / output_steps,
    }


def _median_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = (
        "throughput_tokens_per_s",
        "service_step_ms",
        "build_batch_ms",
        "inter_executor_gap_ms",
        "executor_wall_ms",
        "executor_cuda_event_ms",
    )
    summary = {key: _median([float(run[key]) for run in runs]) for key in scalar_keys}
    summary["build_batch_quartile_ms"] = [
        _median([float(run["build_batch_quartile_ms"][index]) for run in runs])
        for index in range(4)
    ]
    return summary


def _full_summary(
    input_dir: Path, side: str, *, output_steps: int, batch_size: int
) -> dict[str, Any]:
    profile = _load(input_dir / f"{side}-full-r1-stages.json")
    validation = profile["stages"]["execution_request.validate"]
    build = profile["stages"]["engine.build_batch"]
    return {
        "request_validation_ms": float(validation["wall_total_ms"]) / output_steps,
        "request_validation_quartile_ms": [
            float(value) * batch_size for value in validation["wall_quartile_means_ms"]
        ],
        "build_batch_ms": float(build["wall_total_ms"]) / output_steps,
        "build_batch_quartile_ms": [float(value) for value in build["wall_quartile_means_ms"]],
    }


def _relative_change(before: float, after: float) -> float:
    return (after / before - 1.0) * 100.0


def _plot(summary: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    before = summary["median"]["before"]
    after = summary["median"]["after"]
    runs = summary["runs"]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    figure.suptitle("CPU history materialization fix on RTX 5090", fontsize=19)

    colors = ("#9C6B5C", "#2C7FB8")
    labels = ("Before", "After")

    throughput = [before["throughput_tokens_per_s"], after["throughput_tokens_per_s"]]
    axes[0].bar(labels, throughput, color=colors, width=0.58)
    for side_index, side in enumerate(("before", "after")):
        axes[0].scatter(
            [side_index] * len(runs[side]),
            [run["throughput_tokens_per_s"] for run in runs[side]],
            color="#202020",
            zorder=3,
        )
    axes[0].set_ylabel("Output tokens/s")
    axes[0].set_title("Matched steady-decode throughput")
    axes[0].bar_label(axes[0].containers[0], fmt="%.1f", padding=3)

    stage_names = ("Build batch", "Inter-executor gap")
    before_stages = [before["build_batch_ms"], before["inter_executor_gap_ms"]]
    after_stages = [after["build_batch_ms"], after["inter_executor_gap_ms"]]
    positions = range(len(stage_names))
    width = 0.34
    axes[1].bar(
        [position - width / 2 for position in positions],
        before_stages,
        width,
        label="Before",
        color=colors[0],
    )
    axes[1].bar(
        [position + width / 2 for position in positions],
        after_stages,
        width,
        label="After",
        color=colors[1],
    )
    axes[1].set_xticks(list(positions), stage_names)
    axes[1].set_ylabel("Milliseconds per output step")
    axes[1].set_title("CPU gap removed by the patch")
    axes[1].legend()

    quartiles = ("Q1", "Q2", "Q3", "Q4")
    axes[2].plot(
        quartiles,
        before["build_batch_quartile_ms"],
        marker="o",
        linewidth=2.5,
        label="Before",
        color=colors[0],
    )
    axes[2].plot(
        quartiles,
        after["build_batch_quartile_ms"],
        marker="o",
        linewidth=2.5,
        label="After",
        color=colors[1],
    )
    axes[2].set_ylabel("Build ExecutionBatch (ms)")
    axes[2].set_xlabel("Chronological decode quartile")
    axes[2].set_title("Context-length growth is gone")
    axes[2].legend()

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--output-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    runs = {
        side: [
            _lean_run(args.input_dir, side, run, output_steps=args.output_steps)
            for run in range(1, 4)
        ]
        for side in ("before", "after")
    }
    median = {side: _median_summary(side_runs) for side, side_runs in runs.items()}
    before = median["before"]
    after = median["after"]

    before_first = _load(args.input_dir / "before-lean-r1.json")
    before_second = _load(args.input_dir / "before-lean-r2.json")
    after_first = _load(args.input_dir / "after-lean-r1.json")
    output_reproducibility = {
        "before_r1_vs_before_r2_first_mismatch_positions": _first_mismatch_positions(
            before_first, before_second
        ),
        "before_r1_vs_after_r1_first_mismatch_positions": _first_mismatch_positions(
            before_first, after_first
        ),
        "note": (
            "Exact cross-process greedy outputs are not a valid patch gate in this BF16 Triton "
            "run: the unchanged baseline also diverged across repeated runs."
        ),
    }

    summary = {
        "scope": {
            "model": "Qwen2.5-Coder-7B-Instruct",
            "device": "RTX 5090",
            "batch_size": args.batch_size,
            "prompt_tokens_per_request": 16,
            "output_tokens_per_request": args.output_steps,
            "output_steps": args.output_steps,
            "profile": "lean median of 3 paired runs; full single diagnostic pass",
        },
        "runs": runs,
        "median": median,
        "full_profile": {
            side: _full_summary(
                args.input_dir,
                side,
                output_steps=args.output_steps,
                batch_size=args.batch_size,
            )
            for side in ("before", "after")
        },
        "change": {
            "throughput_percent": _relative_change(
                before["throughput_tokens_per_s"], after["throughput_tokens_per_s"]
            ),
            "service_step_ms": after["service_step_ms"] - before["service_step_ms"],
            "build_batch_ms": after["build_batch_ms"] - before["build_batch_ms"],
            "build_batch_percent": _relative_change(
                before["build_batch_ms"], after["build_batch_ms"]
            ),
            "inter_executor_gap_ms": (
                after["inter_executor_gap_ms"] - before["inter_executor_gap_ms"]
            ),
            "executor_cuda_event_percent": _relative_change(
                before["executor_cuda_event_ms"], after["executor_cuda_event_ms"]
            ),
        },
        "output_reproducibility": output_reproducibility,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(summary, args.figure)
    print(json.dumps(summary["change"], indent=2))


if __name__ == "__main__":
    main()
