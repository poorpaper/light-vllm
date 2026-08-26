"""在同一张 GPU、同一组张量上对照 light-vllm 与 vLLM AWQ kernel。"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from light_vllm.modeling.quantization.awq import dequantize_awq, pack_awq
from light_vllm.modeling.quantization.cuda_awq import prepare_cuda_awq_operation


@dataclass(frozen=True, slots=True)
class Measurement:
    implementation: str
    num_tokens: int
    latency_ms_median: float
    latency_ms_min: float
    effective_tflops: float
    peak_temporary_mib: float
    max_abs_error: float


def _time_cuda(call: Callable[[], Tensor], warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return values


def _measure(
    name: str,
    call: Callable[[], Tensor],
    inputs: Tensor,
    out_features: int,
    reference: Tensor,
    warmup: int,
    iterations: int,
) -> Measurement:
    torch.cuda.reset_peak_memory_stats(inputs.device)
    baseline = torch.cuda.memory_allocated(inputs.device)
    output = call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(inputs.device) - baseline
    timings = _time_cuda(call, warmup, iterations)
    median = statistics.median(timings)
    operations = 2 * inputs.shape[0] * inputs.shape[1] * out_features
    return Measurement(
        implementation=name,
        num_tokens=inputs.shape[0],
        latency_ms_median=median,
        latency_ms_min=min(timings),
        effective_tflops=operations / (median / 1000) / 1e12,
        peak_temporary_mib=peak / 1024**2,
        max_abs_error=(output.float() - reference.float()).abs().max().item(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--in-features", type=int, default=3584)
    parser.add_argument("--out-features", type=int, default=3584)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 256])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.in_features % args.group_size or args.out_features % 8:
        raise ValueError("AWQ dimensions must align to group_size and pack factor")

    try:
        from vllm import _custom_ops as vllm_ops
    except ImportError as exc:
        raise RuntimeError("install vLLM in the benchmark environment") from exc

    device = torch.device("cuda")
    dtype = torch.float16
    torch.manual_seed(0)
    natural_weights = torch.randint(
        0,
        16,
        (args.in_features, args.out_features),
        dtype=torch.int8,
        device=device,
    )
    qweight = pack_awq(natural_weights)
    natural_zeros = torch.randint(
        0,
        16,
        (args.in_features // args.group_size, args.out_features),
        dtype=torch.int8,
        device=device,
    )
    qzeros = pack_awq(natural_zeros)
    scales = (
        torch.rand(
            args.in_features // args.group_size,
            args.out_features,
            dtype=dtype,
            device=device,
        )
        * 0.02
    )
    light_operation = prepare_cuda_awq_operation(
        qweight,
        qzeros,
        scales,
        args.group_size,
        None,
    )

    results = []
    for rows in args.tokens:
        inputs = torch.randn(rows, args.in_features, dtype=dtype, device=device)
        reference = inputs @ dequantize_awq(qweight, qzeros, scales, args.group_size)

        def light(current_inputs: Tensor = inputs) -> Tensor:
            return light_operation(current_inputs)

        def vllm(current_inputs: Tensor = inputs, current_rows: int = rows) -> Tensor:
            if current_rows >= 256:
                weight = vllm_ops.awq_dequantize(qweight, scales, qzeros, 0, 0, 0)
                return current_inputs @ weight
            return vllm_ops.awq_gemm(current_inputs, qweight, scales, qzeros, 8)

        for name, call in (("light-vllm-cuda", light), ("vllm-awq", vllm)):
            results.append(
                asdict(
                    _measure(
                        name,
                        call,
                        inputs,
                        args.out_features,
                        reference,
                        args.warmup,
                        args.iterations,
                    )
                )
            )

    payload = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "in_features": args.in_features,
        "out_features": args.out_features,
        "group_size": args.group_size,
        "dtype": "float16",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
