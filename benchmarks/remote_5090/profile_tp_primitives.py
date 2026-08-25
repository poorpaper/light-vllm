"""测量 TP 热路径中的控制通信和 NCCL 集合通信开销。"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from statistics import median
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as functional_collectives

from light_vllm.runtime.execution.interfaces import (
    ExecutionBatch,
    ExecutionOutput,
    ExecutionRequest,
    RequestOutput,
)


@dataclass(frozen=True, slots=True)
class Measurement:
    median_ms: float
    p95_ms: float


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * quantile))
    return ordered[index]


def _measure(
    operation: Any,
    *,
    iterations: int,
    warmup: int,
    synchronize_cuda: bool,
) -> Measurement:
    for _ in range(warmup):
        operation()
    if synchronize_cuda:
        torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        started_at = perf_counter()
        operation()
        if synchronize_cuda:
            torch.cuda.synchronize()
        samples.append((perf_counter() - started_at) * 1000.0)
    return Measurement(median_ms=median(samples), p95_ms=_percentile(samples, 0.95))


def _representative_batch(num_requests: int) -> ExecutionBatch:
    return ExecutionBatch(
        requests=tuple(
            ExecutionRequest(
                request_id=f"request-{index}",
                input_token_ids=(1000 + index,),
                context_token_ids=None,
                num_computed_tokens=255,
                num_lookahead_tokens=0,
                max_output_tokens=1,
                block_ids=tuple(range(index * 16, index * 16 + 16)),
                output_position=255,
            )
            for index in range(num_requests)
        )
    )


def _representative_output(num_requests: int) -> ExecutionOutput:
    return ExecutionOutput(
        requests=tuple(
            RequestOutput(
                request_id=f"request-{index}",
                num_input_tokens_computed=1,
                output_token_ids=(2000 + index,),
            )
            for index in range(num_requests)
        ),
        num_model_tokens_computed=num_requests,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=3584)
    parser.add_argument("--vocab-size", type=int, default=152064)
    parser.add_argument("--num-layer-all-reduces", type=int, default=56)
    parser.add_argument("--high-priority", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0 or args.num_requests <= 0:
        raise ValueError("iteration and request counts must be positive")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    process_group_options = dist.ProcessGroupNCCL.Options(
        is_high_priority_stream=args.high_priority
    )
    dist.init_process_group("nccl", pg_options=process_group_options)
    control_group = dist.new_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)

    batch = _representative_batch(args.num_requests)
    output = _representative_output(args.num_requests)
    hidden = torch.ones(
        (args.num_requests, args.hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    local_vocab_size = (args.vocab_size + world_size - 1) // world_size
    logits = torch.ones(
        (args.num_requests, local_vocab_size),
        dtype=torch.bfloat16,
        device=device,
    )

    def broadcast_batch() -> None:
        values = [batch if rank == 0 else None]
        dist.broadcast_object_list(values, src=0, group=control_group)

    def gather_outputs() -> None:
        values: list[object | None] = [None] * world_size
        dist.all_gather_object(values, output, group=control_group)

    status = torch.zeros(1, dtype=torch.int64)

    def reduce_status() -> None:
        dist.all_reduce(status, op=dist.ReduceOp.MIN, group=control_group)

    def layer_all_reduces() -> None:
        for _ in range(args.num_layer_all_reduces):
            dist.all_reduce(hidden, op=dist.ReduceOp.SUM)

    process_group = dist.group.WORLD

    def direct_layer_all_reduces() -> None:
        for _ in range(args.num_layer_all_reduces):
            process_group.allreduce(hidden, dist.ReduceOp.SUM).wait()

    def functional_layer_all_reduces() -> None:
        current = hidden
        for _ in range(args.num_layer_all_reduces):
            current = functional_collectives.wait_tensor(
                functional_collectives.all_reduce(current, "sum", process_group)
            )

    gathered = [torch.empty_like(logits) for _ in range(world_size)]

    def gather_vocab() -> None:
        dist.all_gather(gathered, logits)
        torch.cat(gathered, dim=-1)

    measurements = {
        "gloo_broadcast_batch": _measure(
            broadcast_batch,
            iterations=args.iterations,
            warmup=args.warmup,
            synchronize_cuda=False,
        ),
        "gloo_all_gather_output": _measure(
            gather_outputs,
            iterations=args.iterations,
            warmup=args.warmup,
            synchronize_cuda=False,
        ),
        "gloo_reduce_status": _measure(
            reduce_status,
            iterations=args.iterations,
            warmup=args.warmup,
            synchronize_cuda=False,
        ),
        "nccl_layer_all_reduces": _measure(
            layer_all_reduces,
            iterations=args.iterations,
            warmup=args.warmup,
            synchronize_cuda=True,
        ),
        "nccl_direct_layer_all_reduces": _measure(
            direct_layer_all_reduces,
            iterations=args.iterations,
            warmup=args.warmup,
            synchronize_cuda=True,
        ),
        "nccl_functional_layer_all_reduces": _measure(
            functional_layer_all_reduces,
            iterations=args.iterations,
            warmup=args.warmup,
            synchronize_cuda=True,
        ),
        "nccl_vocab_all_gather": _measure(
            gather_vocab,
            iterations=args.iterations,
            warmup=args.warmup,
            synchronize_cuda=True,
        ),
    }
    local_result = {name: asdict(value) for name, value in measurements.items()}
    rank_results: list[dict[str, Any] | None] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(local_result, rank_results, dst=0, group=control_group)
    if rank == 0:
        assert rank_results is not None
        print(
            json.dumps(
                {
                    "world_size": world_size,
                    "num_requests": args.num_requests,
                    "hidden_size": args.hidden_size,
                    "vocab_size": args.vocab_size,
                    "num_layer_all_reduces": args.num_layer_all_reduces,
                    "high_priority": args.high_priority,
                    "ranks": rank_results,
                },
                indent=2,
            )
        )

    dist.destroy_process_group(control_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
