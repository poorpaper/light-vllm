from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

Backend = Literal["light-vllm", "vllm"]


@dataclass(frozen=True, slots=True)
class RequestResult:
    request_id: str
    kind: str
    scheduled_offset_s: float
    sent_offset_s: float
    first_token_offset_s: float | None
    completed_offset_s: float
    prompt_tokens: int
    requested_output_tokens: int
    target_output_tokens: int | None
    output_tokens: int
    generated_token_ids: list[int]
    max_inter_token_gap_s: float | None
    status_code: int | None
    error: str | None

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_offset_s is None:
            return None
        return self.first_token_offset_s - self.sent_offset_s

    @property
    def e2e_s(self) -> float:
        return self.completed_offset_s - self.sent_offset_s

    @property
    def tpot_s(self) -> float | None:
        if self.first_token_offset_s is None or self.output_tokens <= 1:
            return None
        return (self.completed_offset_s - self.first_token_offset_s) / (self.output_tokens - 1)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _arrival_offsets(
    requests: list[dict[str, Any]],
    *,
    mode: str,
    request_rate: float,
    seed: int,
) -> list[float]:
    configured = [request.get("arrival_offset_s") for request in requests]
    if any(value is not None for value in configured):
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0
            for value in configured
        ):
            raise ValueError("arrival_offset_s must be a non-negative number on every request")
        return [float(value) for value in configured]
    count = len(requests)
    if mode == "burst":
        return [0.0] * count
    if request_rate <= 0:
        raise ValueError("request_rate must be positive for poisson arrivals")
    random_source = random.Random(seed)
    offsets: list[float] = []
    current = 0.0
    for _ in range(count):
        offsets.append(current)
        current += random_source.expovariate(request_rate)
    return offsets


async def _read_light_vllm(
    response: httpx.Response,
    *,
    target_output_tokens: int | None,
) -> tuple[list[int], str | None, float | None, float | None]:
    generated: list[int] = []
    event_name: str | None = None
    error: str | None = None
    first_token: float | None = None
    last_token: float | None = None
    max_inter_token_gap: float | None = None
    async for line in response.aiter_lines():
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
            continue
        if not line.startswith("data:"):
            continue
        data = json.loads(line.removeprefix("data:").strip())
        if event_name == "token":
            received_at = time.perf_counter()
            if first_token is None:
                first_token = received_at
            if last_token is not None:
                gap = received_at - last_token
                max_inter_token_gap = max(max_inter_token_gap or 0.0, gap)
            last_token = received_at
            generated.append(int(data["token_id"]))
            if target_output_tokens is not None and len(generated) >= target_output_tokens:
                break
        elif event_name == "error":
            error = f"{data.get('code')}: {data.get('detail')}"
    return generated, error, first_token, max_inter_token_gap


async def _read_vllm(
    response: httpx.Response,
    *,
    target_output_tokens: int | None,
) -> tuple[list[int], str | None, float | None, float | None]:
    generated: list[int] = []
    error: str | None = None
    first_token: float | None = None
    last_token: float | None = None
    max_inter_token_gap: float | None = None
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        raw_data = line.removeprefix("data:").strip()
        if raw_data == "[DONE]":
            break
        data = json.loads(raw_data)
        if "error" in data:
            error = str(data["error"])
            continue
        choices = data.get("choices") or []
        for choice in choices:
            token_ids = [int(token_id) for token_id in (choice.get("token_ids") or [])]
            if token_ids:
                received_at = time.perf_counter()
                if first_token is None:
                    first_token = received_at
                if last_token is not None:
                    gap = received_at - last_token
                    max_inter_token_gap = max(max_inter_token_gap or 0.0, gap)
                last_token = received_at
            if target_output_tokens is not None:
                remaining = target_output_tokens - len(generated)
                generated.extend(token_ids[:remaining])
                if len(generated) >= target_output_tokens:
                    return generated, error, first_token, max_inter_token_gap
            else:
                generated.extend(token_ids)
    return generated, error, first_token, max_inter_token_gap


async def _run_request(
    client: httpx.AsyncClient,
    *,
    backend: Backend,
    base_url: str,
    model: str | None,
    request: dict[str, Any],
    scheduled_offset_s: float,
    benchmark_start: float,
    respect_eos: bool,
    target_output_tokens: int | None,
) -> RequestResult:
    target = benchmark_start + scheduled_offset_s
    await asyncio.sleep(max(0.0, target - time.perf_counter()))
    sent = time.perf_counter()
    status_code: int | None = None
    error: str | None = None
    generated: list[int] = []
    first_token: float | None = None
    max_inter_token_gap: float | None = None

    if backend == "light-vllm":
        url = f"{base_url.rstrip('/')}/generate/stream"
        payload = {
            "input_ids": request["input_ids"],
            "max_new_tokens": request["max_new_tokens"],
            "eos_token_id": request.get("eos_token_id") if respect_eos else None,
        }
        if respect_eos and payload["eos_token_id"] is None:
            raise ValueError("--respect-eos requires eos_token_id on every request")
    else:
        if model is None:
            raise ValueError("--model is required for vLLM")
        url = f"{base_url.rstrip('/')}/v1/completions"
        payload = {
            "model": model,
            "prompt": request["input_ids"],
            "max_tokens": request["max_new_tokens"],
            "temperature": 0.0,
            "ignore_eos": not respect_eos,
            "stream": True,
            "return_token_ids": True,
        }

    try:
        async with client.stream("POST", url, json=payload) as response:
            status_code = response.status_code
            if response.status_code != 200:
                error = (await response.aread()).decode("utf-8", errors="replace")
            elif backend == "light-vllm":
                generated, error, first_token, max_inter_token_gap = await _read_light_vllm(
                    response,
                    target_output_tokens=target_output_tokens,
                )
            else:
                generated, error, first_token, max_inter_token_gap = await _read_vllm(
                    response,
                    target_output_tokens=target_output_tokens,
                )
    except Exception as exc:  # benchmark must retain failures as evidence
        error = f"{type(exc).__name__}: {exc}"

    completed = time.perf_counter()
    return RequestResult(
        request_id=str(request["id"]),
        kind=str(request.get("kind", "unknown")),
        scheduled_offset_s=scheduled_offset_s,
        sent_offset_s=sent - benchmark_start,
        first_token_offset_s=(first_token - benchmark_start if first_token else None),
        completed_offset_s=completed - benchmark_start,
        prompt_tokens=len(request["input_ids"]),
        requested_output_tokens=int(request["max_new_tokens"]),
        target_output_tokens=target_output_tokens,
        output_tokens=len(generated),
        generated_token_ids=generated,
        max_inter_token_gap_s=max_inter_token_gap,
        status_code=status_code,
        error=error,
    )


async def _fetch_text(client: httpx.AsyncClient, url: str) -> str | None:
    try:
        response = await client.get(url)
        response.raise_for_status()
        return response.text
    except Exception:
        return None


def _metric_summary(results: list[RequestResult], elapsed_s: float) -> dict[str, object]:
    successful = [result for result in results if result.error is None]
    ttft = [value for result in successful if (value := result.ttft_s) is not None]
    tpot = [value for result in successful if (value := result.tpot_s) is not None]
    max_inter_token_gaps = [
        value for result in successful if (value := result.max_inter_token_gap_s) is not None
    ]
    e2e = [result.e2e_s for result in successful]
    output_tokens = sum(result.output_tokens for result in successful)
    output_lengths = [result.output_tokens for result in successful]
    return {
        "requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "elapsed_s": elapsed_s,
        "output_tokens": output_tokens,
        "output_tokens_per_s": output_tokens / elapsed_s if elapsed_s else 0.0,
        "output_tokens_mean": statistics.fmean(output_lengths) if output_lengths else None,
        "output_tokens_p50": _percentile(output_lengths, 0.5),
        "output_tokens_p95": _percentile(output_lengths, 0.95),
        "output_tokens_max": max(output_lengths, default=None),
        "output_limit_hits": sum(
            result.output_tokens >= result.requested_output_tokens for result in successful
        ),
        "replay_target_hits": sum(
            result.target_output_tokens is not None
            and result.output_tokens >= result.target_output_tokens
            for result in successful
        ),
        "ttft_p50_s": _percentile(ttft, 0.5),
        "ttft_p95_s": _percentile(ttft, 0.95),
        "ttft_p99_s": _percentile(ttft, 0.99),
        "tpot_p50_s": _percentile(tpot, 0.5),
        "tpot_p95_s": _percentile(tpot, 0.95),
        "max_itl_p50_s": _percentile(max_inter_token_gaps, 0.5),
        "max_itl_p95_s": _percentile(max_inter_token_gaps, 0.95),
        "max_itl_p99_s": _percentile(max_inter_token_gaps, 0.99),
        "max_itl_max_s": max(max_inter_token_gaps, default=None),
        "e2e_p50_s": _percentile(e2e, 0.5),
        "e2e_p95_s": _percentile(e2e, 0.95),
        "e2e_p99_s": _percentile(e2e, 0.99),
        "e2e_mean_s": statistics.fmean(e2e) if e2e else None,
    }


def _summary(results: list[RequestResult], elapsed_s: float) -> dict[str, object]:
    by_kind = {
        kind: _metric_summary(
            [result for result in results if result.kind == kind],
            elapsed_s,
        )
        for kind in sorted({result.kind for result in results})
    }
    return {
        **_metric_summary(results, elapsed_s),
        "by_kind": by_kind,
    }


async def _main_async(args: argparse.Namespace) -> None:
    workload = json.loads(args.workload.read_text(encoding="utf-8"))
    requests: list[dict[str, Any]] = workload["requests"]
    if args.limit is not None:
        requests = requests[: args.limit]
    offsets = _arrival_offsets(
        requests,
        mode=args.arrival_mode,
        request_rate=args.request_rate,
        seed=args.seed,
    )
    replay_lengths: dict[str, int] | None = None
    if args.replay_lengths_from is not None:
        replay_payload = json.loads(args.replay_lengths_from.read_text(encoding="utf-8"))
        replay_lengths = {
            str(result["request_id"]): int(result["output_tokens"])
            for result in replay_payload["results"]
        }
        missing = [
            str(request["id"]) for request in requests if str(request["id"]) not in replay_lengths
        ]
        if missing:
            raise ValueError(f"replay result is missing {len(missing)} request IDs")
        invalid = [
            str(request["id"])
            for request in requests
            if not 0 < replay_lengths[str(request["id"])] <= int(request["max_new_tokens"])
        ]
        if invalid:
            raise ValueError(f"replay result has invalid lengths for {len(invalid)} requests")

    limits = httpx.Limits(
        max_connections=args.max_connections,
        max_keepalive_connections=(0 if args.disable_keepalive else args.max_connections),
    )
    timeout = httpx.Timeout(args.timeout)
    headers = {"Connection": "close"} if args.disable_keepalive else None
    async with httpx.AsyncClient(timeout=timeout, limits=limits, headers=headers) as client:
        metrics_url = f"{args.base_url.rstrip('/')}/metrics"
        metrics_before = await _fetch_text(client, metrics_url)
        start = time.perf_counter()
        tasks = [
            asyncio.create_task(
                _run_request(
                    client,
                    backend=args.backend,
                    base_url=args.base_url,
                    model=args.model,
                    request=request,
                    scheduled_offset_s=offset,
                    benchmark_start=start,
                    respect_eos=args.respect_eos,
                    target_output_tokens=(
                        replay_lengths[str(request["id"])] if replay_lengths is not None else None
                    ),
                )
            )
            for request, offset in zip(requests, offsets, strict=True)
        ]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start
        metrics_after = await _fetch_text(client, metrics_url)

    payload = {
        "metadata": {
            "case": args.case,
            "backend": args.backend,
            "model": args.model,
            "workload": workload["name"],
            "arrival_mode": args.arrival_mode,
            "request_rate": args.request_rate,
            "seed": args.seed,
            "respect_eos": args.respect_eos,
            "replay_lengths_from": (
                args.replay_lengths_from.name if args.replay_lengths_from is not None else None
            ),
        },
        "summary": _summary(results, elapsed),
        "results": [
            {
                **asdict(result),
                "ttft_s": result.ttft_s,
                "tpot_s": result.tpot_s,
                "e2e_s": result.e2e_s,
            }
            for result in results
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if metrics_before is not None:
        args.output.with_suffix(".metrics-before.prom").write_text(metrics_before, encoding="utf-8")
    if metrics_after is not None:
        args.output.with_suffix(".metrics-after.prom").write_text(metrics_after, encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("light-vllm", "vllm"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--arrival-mode", choices=("burst", "poisson"), default="burst")
    parser.add_argument("--request-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--max-connections", type=int, default=128)
    parser.add_argument(
        "--disable-keepalive",
        action="store_true",
        help="open fresh TCP connections so a Kubernetes Service can reach new replicas",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--respect-eos",
        action="store_true",
        help="use each request's EOS token instead of forcing the configured maximum",
    )
    parser.add_argument(
        "--replay-lengths-from",
        type=Path,
        help="stop each stream at the matching observed output length from a prior result",
    )
    args = parser.parse_args()
    if args.respect_eos and args.replay_lengths_from is not None:
        parser.error("--respect-eos and --replay-lengths-from are mutually exclusive")
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
