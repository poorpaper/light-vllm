"""创建并启动 HTTP 服务。

这里负责选择具体实现、加载模型、解析命令行参数并启动 Uvicorn，
不放推理和 HTTP 编码逻辑。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from light_vllm.bootstrap import create_runner
from light_vllm.modeling.models.interfaces import ModelSpec
from light_vllm.runtime.engine.core import EngineCore
from light_vllm.runtime.engine.in_process import InProcessEngineClient
from light_vllm.runtime.engine.interfaces import EngineClient
from light_vllm.runtime.execution.interfaces import ExecutionTimer
from light_vllm.runtime.execution.local import LocalModelExecutor, LocalTokenExecutor
from light_vllm.runtime.execution.paged_attention import (
    PagedAttentionBackend,
    TorchPagedAttentionBackend,
)
from light_vllm.runtime.execution.paged_cache import (
    CudaMemoryKVCachePlanner,
    PagedKVCacheConfig,
    PagedKVCachePlanner,
)
from light_vllm.runtime.execution.speculative import (
    GreedyAcceptanceSampler,
    NGramSpeculativeDecodeHandler,
    NGramTokenProposer,
)
from light_vllm.runtime.execution.timing import (
    CudaEventExecutionTimer,
    WallClockExecutionTimer,
)
from light_vllm.runtime.execution.worker import (
    ContiguousStepHandler,
    LocalModelWorker,
    PagedStepHandler,
    StandardDecodeHandler,
)
from light_vllm.runtime.generation.reference import ReferenceGenerationService
from light_vllm.runtime.kv_cache import (
    ContiguousKVCacheConfig,
    PagedKVCacheManager,
    UnboundedKVCacheManager,
)
from light_vllm.runtime.observability.interfaces import PerformanceMetricsReader
from light_vllm.runtime.observability.performance import InMemoryPerformanceObserver
from light_vllm.runtime.sampling import GreedySampler
from light_vllm.runtime.scheduler.interfaces import DecodingBudget, ShortRequestPolicy
from light_vllm.runtime.scheduler.token_budget import TokenBudgetScheduler

if TYPE_CHECKING:
    from fastapi import FastAPI


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
RuntimeMode = Literal["reference", "engine"]
KVReservationMode = Literal["blocks", "unbounded"]
PagedAttentionBackendName = Literal["torch", "triton"]


def _create_paged_cache_planner(
    spec: ModelSpec,
    *,
    num_blocks: int | None,
    block_size: int,
    memory_fraction: float,
) -> PagedKVCachePlanner:
    if num_blocks is not None:
        # CPU 无法发现显存容量，因此测试时直接指定固定页数。
        return PagedKVCacheConfig(
            num_blocks=num_blocks,
            block_size=block_size,
            dtype=spec.dtype,
            device=spec.device,
        )
    if torch.device(spec.device).type == "cuda":
        return CudaMemoryKVCachePlanner(
            block_size=block_size,
            dtype=spec.dtype,
            device=spec.device,
            memory_fraction=memory_fraction,
        )
    raise ValueError("CPU paged execution requires an explicit num_kv_blocks")


def _create_paged_attention_backend(
    name: PagedAttentionBackendName,
    spec: ModelSpec,
) -> PagedAttentionBackend:
    if name == "torch":
        return TorchPagedAttentionBackend()
    if name != "triton":
        raise ValueError(f"unsupported paged attention backend: {name}")
    if torch.device(spec.device).type != "cuda":
        raise ValueError("Triton paged attention requires a CUDA device")

    # Triton 是可选依赖；默认 PyTorch 路径不会导入它。
    from light_vllm.runtime.execution.triton_paged_attention import (
        TritonPagedAttentionBackend,
    )

    return TritonPagedAttentionBackend()


def _create_execution_timer(spec: ModelSpec) -> ExecutionTimer:
    if torch.device(spec.device).type == "cuda":
        return CudaEventExecutionTimer(spec.device)
    return WallClockExecutionTimer()


def create_serving_app(
    spec: ModelSpec,
    *,
    runtime: RuntimeMode = "reference",
    kv_reservation: KVReservationMode = "blocks",
    paged_attention_backend: PagedAttentionBackendName = "torch",
    max_num_sequences: int = 8,
    max_num_scheduled_tokens: int = 256,
    num_kv_blocks: int | None = None,
    kv_block_size: int = 16,
    kv_cache_memory_fraction: float = 0.8,
    enable_prefix_caching: bool = False,
    num_speculative_tokens: int = 0,
    speculative_ngram_min: int = 2,
    speculative_ngram_max: int = 5,
    short_request_policy: ShortRequestPolicy | None = None,
) -> FastAPI:
    """创建单进程 HTTP 服务，并选择 reference 或 Engine Core。

    以后换成独立进程引擎时只改这里，不改 HTTP 路由。
    """

    # FastAPI 是可选依赖。只有启动 HTTP 服务时才导入，
    # 没有安装它也不影响核心模型功能。
    from light_vllm.serving.http import create_http_app

    runner = create_runner()
    # reference client 没有常驻 driver；只有批量 Engine 需要在 lifespan
    # 结束时显式等待当前模型迭代完成。
    close_engine: Callable[[], Awaitable[None]] | None = None
    initialize_executor: Callable[[], None] | None = None
    refresh_performance_metrics: Callable[[], None] | None = None
    performance_metrics: PerformanceMetricsReader | None = None
    sampler = GreedySampler()
    if type(num_speculative_tokens) is not int or num_speculative_tokens < 0:
        raise ValueError("num_speculative_tokens must be a non-negative integer")
    if paged_attention_backend not in ("torch", "triton"):
        raise ValueError(f"unsupported paged attention backend: {paged_attention_backend}")
    if runtime == "reference":
        if short_request_policy is not None:
            raise ValueError("short-request scheduling requires the engine runtime")
        if num_speculative_tokens:
            raise ValueError("speculative decoding requires the engine runtime")
        if paged_attention_backend != "torch":
            raise ValueError("paged attention backend selection requires the engine runtime")
        # 保留原始单请求基线，继续通过轻量 sync-to-async bridge 对外服务。
        executor = LocalTokenExecutor(runner, sampler, device=spec.device)
        service = ReferenceGenerationService(executor)
        engine: EngineClient = InProcessEngineClient(service)
    else:
        if runtime != "engine":
            raise ValueError(f"unsupported runtime mode: {runtime}")
        if kv_reservation == "blocks":
            cache_planner = _create_paged_cache_planner(
                spec,
                num_blocks=num_kv_blocks,
                block_size=kv_block_size,
                memory_fraction=kv_cache_memory_fraction,
            )
            # 同一容量对象同时交给逻辑分配和物理页池，避免两份配置漂移。
            logical_cache = PagedKVCacheManager(
                cache_planner,
                enable_prefix_caching=enable_prefix_caching,
            )
            step_factory = partial(
                PagedStepHandler,
                cache_planner=cache_planner,
                attention_backend=_create_paged_attention_backend(
                    paged_attention_backend,
                    spec,
                ),
            )
        elif kv_reservation == "unbounded":
            if enable_prefix_caching:
                raise ValueError("prefix caching requires paged KV reservation")
            if paged_attention_backend != "torch":
                raise ValueError("Triton paged attention requires paged KV reservation")
            # 连续缓存也直接读取模型声明的 KV 形状，这里只指定设备和数据类型。
            logical_cache = UnboundedKVCacheManager()
            step_factory = partial(
                ContiguousStepHandler,
                cache_config=ContiguousKVCacheConfig(
                    dtype=spec.dtype,
                    device=spec.device,
                ),
            )
        else:
            raise ValueError(f"unsupported KV reservation mode: {kv_reservation}")
        if num_speculative_tokens:
            decode_handler = NGramSpeculativeDecodeHandler(
                NGramTokenProposer(
                    min_match_length=speculative_ngram_min,
                    max_match_length=speculative_ngram_max,
                ),
                sampler,
                GreedyAcceptanceSampler(),
            )
            decoding_budget = DecodingBudget(
                num_lookahead_tokens=num_speculative_tokens,
                max_output_tokens=num_speculative_tokens + 1,
            )
        else:
            decode_handler = StandardDecodeHandler(sampler)
            decoding_budget = None
        worker = LocalModelWorker(
            runner,
            step_factory,
            decode_handler,
        )
        model_executor = LocalModelExecutor(
            worker,
            timer=_create_execution_timer(spec),
        )
        scheduler = TokenBudgetScheduler(
            logical_cache,
            max_num_sequences=max_num_sequences,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            decoding_budget=decoding_budget,
            short_request_policy=short_request_policy,
        )
        performance_observer = InMemoryPerformanceObserver(spec.architecture)
        engine_core = EngineCore(
            model_executor,
            scheduler,
            performance_observer=performance_observer,
        )
        engine = engine_core
        close_engine = engine_core.close
        initialize_executor = model_executor.initialize
        refresh_performance_metrics = engine_core.refresh_performance_metrics
        performance_metrics = performance_observer

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # 模型加载成功后服务才会就绪；加载失败就停止启动。
        runner.load(spec)
        if initialize_executor is not None:
            initialize_executor()
        if refresh_performance_metrics is not None:
            # CUDA KV 容量直到 executor 初始化后才确定，此时发布首个真实快照。
            refresh_performance_metrics()
        try:
            yield
        finally:
            # close 会先拒绝新请求，再等待正在运行的同步模型步骤安全结束。
            if close_engine is not None:
                await close_engine()

    return create_http_app(
        engine,
        lifespan=lifespan,
        performance_metrics=performance_metrics,
    )


def _json_object(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("model args must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("model args must be a JSON object")
    return parsed


def _create_short_request_policy(
    *,
    max_effective_prompt_tokens: int | None,
    max_total_tokens: int | None,
    reserved_scheduled_tokens: int | None,
    reserved_kv_token_slots: int | None,
    reserved_sequences: int,
    regular_aging_steps: int,
) -> ShortRequestPolicy | None:
    if max_effective_prompt_tokens is None:
        if any(
            value is not None
            for value in (
                max_total_tokens,
                reserved_scheduled_tokens,
                reserved_kv_token_slots,
            )
        ):
            raise ValueError("short-request max effective prompt tokens must also be set")
        return None
    if max_total_tokens is None or reserved_scheduled_tokens is None:
        raise ValueError("short-request token limits and scheduled reserve must all be set")
    if reserved_kv_token_slots is None:
        raise ValueError("short-request KV reserve must be set")
    return ShortRequestPolicy(
        max_effective_prompt_tokens=max_effective_prompt_tokens,
        max_total_tokens=max_total_tokens,
        reserved_scheduled_tokens=reserved_scheduled_tokens,
        reserved_kv_token_slots=reserved_kv_token_slots,
        reserved_sequences=reserved_sequences,
        regular_aging_steps=regular_aging_steps,
    )


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve light-vllm over HTTP")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--architecture", default="tiny-causal-lm")
    parser.add_argument("--loader", default="init")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="float32")
    parser.add_argument("--model-args", type=_json_object, default={})
    parser.add_argument("--runtime", choices=("reference", "engine"), default="reference")
    parser.add_argument(
        "--kv-reservation",
        choices=("blocks", "unbounded"),
        default="blocks",
        help="use physical paged KV or a contiguous unbounded experiment baseline",
    )
    parser.add_argument("--max-num-sequences", type=int, default=8)
    parser.add_argument("--max-num-scheduled-tokens", type=int, default=256)
    parser.add_argument(
        "--num-kv-blocks",
        type=int,
        help="fixed paged-KV capacity; required for CPU correctness mode",
    )
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--kv-cache-memory-fraction", type=float, default=0.8)
    parser.add_argument(
        "--paged-attention-backend",
        choices=("torch", "triton"),
        default="torch",
        help="paged attention implementation used by the engine runtime",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="reuse complete prompt KV blocks across requests",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=0,
        help="maximum n-gram draft tokens per engine step",
    )
    parser.add_argument("--speculative-ngram-min", type=int, default=2)
    parser.add_argument("--speculative-ngram-max", type=int, default=5)
    short = parser.add_argument_group("short-request scheduling")
    short.add_argument("--short-request-max-effective-prompt-tokens", type=int)
    short.add_argument("--short-request-max-total-tokens", type=int)
    short.add_argument("--short-request-reserved-scheduled-tokens", type=int)
    short.add_argument("--short-request-reserved-kv-token-slots", type=int)
    short.add_argument("--short-request-reserved-sequences", type=int, default=1)
    short.add_argument("--regular-request-aging-steps", type=int, default=8)
    return parser


def main() -> None:
    import uvicorn

    args = _create_parser().parse_args()
    spec = ModelSpec(
        architecture=args.architecture,
        loader=args.loader,
        model_args=args.model_args,
        weights=args.weights,
        device=args.device,
        dtype=_DTYPES[args.dtype],
    )
    uvicorn.run(
        create_serving_app(
            spec,
            runtime=args.runtime,
            kv_reservation=args.kv_reservation,
            paged_attention_backend=args.paged_attention_backend,
            max_num_sequences=args.max_num_sequences,
            max_num_scheduled_tokens=args.max_num_scheduled_tokens,
            num_kv_blocks=args.num_kv_blocks,
            kv_block_size=args.kv_block_size,
            kv_cache_memory_fraction=args.kv_cache_memory_fraction,
            enable_prefix_caching=args.enable_prefix_caching,
            num_speculative_tokens=args.num_speculative_tokens,
            speculative_ngram_min=args.speculative_ngram_min,
            speculative_ngram_max=args.speculative_ngram_max,
            short_request_policy=_create_short_request_policy(
                max_effective_prompt_tokens=(args.short_request_max_effective_prompt_tokens),
                max_total_tokens=args.short_request_max_total_tokens,
                reserved_scheduled_tokens=(args.short_request_reserved_scheduled_tokens),
                reserved_kv_token_slots=args.short_request_reserved_kv_token_slots,
                reserved_sequences=args.short_request_reserved_sequences,
                regular_aging_steps=args.regular_request_aging_steps,
            ),
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
