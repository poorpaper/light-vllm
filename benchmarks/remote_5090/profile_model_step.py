from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from light_vllm.bootstrap import create_runner
from light_vllm.modeling.models.interfaces import ModelSpec
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
    logits_by_request = handler.forward(session, batch)
    if sampler is not None:
        sampler.sample(torch.stack([logits[-1] for logits in logits_by_request]))


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
    args = parser.parse_args()

    device = torch.device("cuda:0")
    spec = ModelSpec(
        architecture="qwen2",
        loader="safetensors",
        weights=args.weights,
        device=device,
        dtype=torch.bfloat16,
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
    trace_path = args.output.with_suffix(".trace.json")
    profile.export_chrome_trace(str(trace_path))
    payload = {
        "batch_size": args.batch_size,
        "computed_tokens": args.computed_tokens,
        "query_tokens": args.query_tokens,
        "timed_steps": args.timed_steps,
        "profile_steps": args.profile_steps,
        "include_greedy_sampling": args.include_greedy_sampling,
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
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "trace"}, indent=2))


if __name__ == "__main__":
    main()
