from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from light_vllm.bootstrap import create_runner
from light_vllm.modeling.models.interfaces import ModelSpec
from light_vllm.modeling.tensor_parallel import TensorParallelContext
from light_vllm.runtime.execution.distributed import TorchDistributedGroup
from light_vllm.runtime.execution.interfaces import ModelStepBatch, ModelStepRequest
from light_vllm.runtime.execution.layout import linear_query_layout
from light_vllm.runtime.execution.paged_cache import PagedKVCacheConfig
from light_vllm.runtime.execution.triton_paged_attention import TritonPagedAttentionBackend
from light_vllm.runtime.execution.worker import PagedStepHandler
from light_vllm.runtime.sampling import GreedySampler


def _batch(
    *,
    batch_size: int,
    computed_tokens: int,
    query_tokens: int,
    block_size: int,
) -> ModelStepBatch:
    blocks_per_request = (computed_tokens + query_tokens + block_size - 1) // block_size
    requests = []
    for row in range(batch_size):
        first_block = row * blocks_per_request
        requests.append(
            ModelStepRequest(
                request_id=f"profile-{row}",
                query_token_ids=(1,) * query_tokens,
                num_computed_tokens=computed_tokens,
                num_reserved_query_tokens=query_tokens,
                query_layout=linear_query_layout(query_tokens),
                block_ids=tuple(range(first_block, first_block + blocks_per_request)),
            )
        )
    return ModelStepBatch(tuple(requests))


def _execute_step(
    handler: PagedStepHandler,
    session,
    batch: ModelStepBatch,
    *,
    sampler: GreedySampler | None,
) -> None:
    output = handler.forward(session, batch)
    if sampler is not None:
        sampler.sample(output.logits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--computed-tokens", type=int, default=256)
    parser.add_argument("--query-tokens", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=1)
    parser.add_argument("--include-greedy-sampling", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    if args.tensor_parallel_size <= 0:
        raise ValueError("tensor parallel size must be positive")
    launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if launched_world_size != args.tensor_parallel_size:
        raise ValueError("launch world size must match --tensor-parallel-size")

    group = TorchDistributedGroup.initialize() if args.tensor_parallel_size > 1 else None
    rank = 0 if group is None else group.rank
    device = torch.device("cuda:0") if group is None else group.device
    parallel = None if group is None else TensorParallelContext(group.rank, group.world_size, group)
    spec = ModelSpec(
        architecture="qwen2",
        loader="safetensors",
        weights=args.weights,
        device=device,
        dtype=torch.bfloat16,
        tensor_parallel=parallel,
    )
    runner = create_runner()
    runner.load(spec)
    session = runner.open_session()
    batch = _batch(
        batch_size=args.batch_size,
        computed_tokens=args.computed_tokens,
        query_tokens=args.query_tokens,
        block_size=args.block_size,
    )
    blocks_per_request = (
        args.computed_tokens + args.query_tokens + args.block_size - 1
    ) // args.block_size
    handler = PagedStepHandler(
        session,
        PagedKVCacheConfig(
            num_blocks=args.batch_size * blocks_per_request,
            block_size=args.block_size,
            dtype=torch.bfloat16,
            device=device,
        ),
        TritonPagedAttentionBackend(),
    )
    sampler = GreedySampler() if args.include_greedy_sampling else None

    for _ in range(args.warmup_steps):
        _execute_step(handler, session, batch, sampler=sampler)
    torch.cuda.synchronize(device)

    started = time.perf_counter()
    for _ in range(args.timed_steps):
        _execute_step(handler, session, batch, sampler=sampler)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        record_shapes=True,
    ) as profile:
        for _ in range(args.profile_steps):
            _execute_step(handler, session, batch, sampler=sampler)
        torch.cuda.synchronize(device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rank_output = (
        args.output
        if group is None
        else args.output.with_name(f"{args.output.stem}.rank{rank}{args.output.suffix}")
    )
    trace_path = rank_output.with_suffix(".trace.json")
    profile.export_chrome_trace(str(trace_path))
    payload = {
        "batch_size": args.batch_size,
        "computed_tokens": args.computed_tokens,
        "query_tokens": args.query_tokens,
        "timed_steps": args.timed_steps,
        "profile_steps": args.profile_steps,
        "include_greedy_sampling": args.include_greedy_sampling,
        "tensor_parallel_size": args.tensor_parallel_size,
        "rank": rank,
        "elapsed_s": elapsed,
        "mean_step_ms": elapsed * 1000 / args.timed_steps,
        "profiler_cuda": profile.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=40,
        ),
        "profiler_cpu": profile.key_averages().table(
            sort_by="self_cpu_time_total",
            row_limit=40,
        ),
        "trace": str(trace_path),
    }
    rank_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = {
        "rank": rank,
        "mean_step_ms": payload["mean_step_ms"],
        "profile": str(rank_output),
        "trace": str(trace_path),
    }
    if group is None:
        print(
            json.dumps(
                {key: value for key, value in payload.items() if key != "trace"},
                indent=2,
            )
        )
    else:
        rank_summaries = group.all_gather_object(summary)
        if rank == 0:
            combined = {
                "tensor_parallel_size": args.tensor_parallel_size,
                "batch_size": args.batch_size,
                "computed_tokens": args.computed_tokens,
                "query_tokens": args.query_tokens,
                "ranks": rank_summaries,
            }
            args.output.write_text(json.dumps(combined, indent=2), encoding="utf-8")
            print(json.dumps(combined, indent=2))
        group.close()


if __name__ == "__main__":
    main()
