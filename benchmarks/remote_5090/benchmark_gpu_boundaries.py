from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_us": statistics.fmean(values),
        "p50_us": _percentile(values, 0.50),
        "p95_us": _percentile(values, 0.95),
    }


def _measure_cuda(
    operation: Callable[[], Tensor | tuple[Tensor, ...]],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, dict[str, float]]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    event_us: list[float] = []
    wall_us: list[float] = []
    for _ in range(iterations):
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter_ns()
        started.record()
        operation()
        finished.record()
        finished.synchronize()
        wall_us.append((time.perf_counter_ns() - wall_started) / 1_000)
        event_us.append(started.elapsed_time(finished) * 1_000)
    return {"cuda_event": _summarize(event_us), "wall": _summarize(wall_us)}


def _sampler_cases(
    *,
    rows: int,
    vocabulary: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    logits = torch.randn((rows, vocabulary), dtype=dtype, device=device)
    logits_by_request = tuple(logits[row : row + 1] for row in range(rows))

    def direct() -> Tensor:
        return logits.argmax(dim=-1)

    def split_stack() -> Tensor:
        sample_logits = torch.stack([request_logits[-1] for request_logits in logits_by_request])
        return sample_logits.argmax(dim=-1)

    return {
        "rows": rows,
        "vocabulary": vocabulary,
        "direct_argmax": _measure_cuda(direct, warmup=warmup, iterations=iterations),
        "split_stack_argmax": _measure_cuda(
            split_stack,
            warmup=warmup,
            iterations=iterations,
        ),
    }


def _metadata_cases(
    *,
    batch_size: int,
    max_blocks: int,
    query_length: int,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    block_tables = tuple(tuple(range(max_blocks)) for _ in range(batch_size))
    computed = tuple(max_blocks * 16 - query_length for _ in range(batch_size))
    query_lengths = (query_length,) * batch_size
    query_start_loc = tuple(index * query_length for index in range(batch_size + 1))
    request_indices = tuple(
        request_index for request_index in range(batch_size) for _ in range(query_length)
    )
    packed = tuple(value for table in block_tables for value in table)
    packed += computed + query_lengths + query_start_loc + request_indices

    def separate() -> tuple[Tensor, ...]:
        return (
            torch.tensor(block_tables, dtype=torch.int32, device=device),
            torch.tensor(computed, dtype=torch.int32, device=device),
            torch.tensor(query_lengths, dtype=torch.int32, device=device),
            torch.tensor(query_start_loc, dtype=torch.int32, device=device),
            torch.tensor(request_indices, dtype=torch.int32, device=device),
        )

    def single_packed() -> Tensor:
        return torch.tensor(packed, dtype=torch.int32, device=device)

    return {
        "batch_size": batch_size,
        "max_blocks": max_blocks,
        "query_length": query_length,
        "separate_tensors": _measure_cuda(
            separate,
            warmup=warmup,
            iterations=iterations,
        ),
        "single_packed_tensor": _measure_cuda(
            single_packed,
            warmup=warmup,
            iterations=iterations,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--vocabulary", type=int, default=152_064)
    parser.add_argument("--max-blocks", type=int, default=33)
    parser.add_argument("--query-length", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1_000)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    result = {
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "sampler": _sampler_cases(
            rows=args.rows,
            vocabulary=args.vocabulary,
            device=device,
            dtype=torch.bfloat16,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        "metadata": _metadata_cases(
            batch_size=args.rows,
            max_blocks=args.max_blocks,
            query_length=args.query_length,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
