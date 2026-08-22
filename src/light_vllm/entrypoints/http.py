"""创建并启动 HTTP 服务。

这里负责选择具体实现、加载模型、解析命令行参数并启动 Uvicorn，
不放推理和 HTTP 编码逻辑。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from light_vllm.bootstrap import create_runner
from light_vllm.modeling.models.interfaces import ModelSpec
from light_vllm.runtime.engine.admission import (
    PredictiveTTFTAdmission,
    SlidingWindowStepLatencyPredictor,
)
from light_vllm.runtime.engine.core import EngineCore
from light_vllm.runtime.engine.in_process import InProcessEngineClient
from light_vllm.runtime.engine.interfaces import EngineClient
from light_vllm.runtime.engine.process import EngineProcessRuntime, ProcessEngineClient
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
    GreedyTreeAcceptanceSampler,
    NGramChainProposer,
    NGramTrieProposer,
    SpeculativeDecodeHandler,
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
from light_vllm.runtime.scheduler.interfaces import (
    DecodingBudget,
    SelfResubmitPolicy,
    ShortRequestPolicy,
)
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
SpeculativeProposerName = Literal["chain", "trie"]


@dataclass(frozen=True, slots=True)
class _EngineRuntimeConfig:
    kv_reservation: KVReservationMode
    paged_attention_backend: PagedAttentionBackendName
    max_num_sequences: int
    max_num_scheduled_tokens: int
    num_kv_blocks: int | None
    kv_block_size: int
    kv_cache_memory_fraction: float
    enable_prefix_caching: bool
    num_speculative_tokens: int
    speculative_proposer: SpeculativeProposerName
    speculative_ngram_min: int
    speculative_ngram_max: int
    speculative_max_depth: int
    speculative_max_branching: int
    short_request_policy: ShortRequestPolicy | None
    max_tolerable_ttft_seconds: float | None
    ttft_prediction_window_size: int
    ttft_prediction_min_observations: int
    ttft_prediction_quantile: float
    max_pending_requests: int | None
    ttft_kv_cache_watermark: float | None
    enable_self_resubmit: bool
    max_self_resubmits: int
    self_resubmit_strict_fallback_rolled_back_tokens: int
    self_resubmit_initial_extra_blocks: int
    self_resubmit_kv_admission_watermark: float


@dataclass(frozen=True, slots=True)
class _EngineRuntime:
    engine: EngineCore
    performance_metrics: InMemoryPerformanceObserver
    start: Callable[[], None]


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


def _create_engine_runtime(
    spec: ModelSpec,
    config: _EngineRuntimeConfig,
    *,
    execute_inline: bool = False,
) -> _EngineRuntime:
    """装配一个 Engine；模型与 CUDA 资源留到 ``start`` 再初始化。"""

    runner = create_runner()
    performance_observer = InMemoryPerformanceObserver(spec.architecture)
    if config.kv_reservation == "blocks":
        cache_planner = _create_paged_cache_planner(
            spec,
            num_blocks=config.num_kv_blocks,
            block_size=config.kv_block_size,
            memory_fraction=config.kv_cache_memory_fraction,
        )
        logical_cache = PagedKVCacheManager(
            cache_planner,
            enable_prefix_caching=(config.enable_prefix_caching or config.enable_self_resubmit),
        )
        step_factory = partial(
            PagedStepHandler,
            cache_planner=cache_planner,
            attention_backend=_create_paged_attention_backend(
                config.paged_attention_backend,
                spec,
            ),
        )
    elif config.kv_reservation == "unbounded":
        if config.enable_prefix_caching:
            raise ValueError("prefix caching requires paged KV reservation")
        if config.paged_attention_backend != "torch":
            raise ValueError("Triton paged attention requires paged KV reservation")
        logical_cache = UnboundedKVCacheManager()
        step_factory = partial(
            ContiguousStepHandler,
            cache_config=ContiguousKVCacheConfig(
                dtype=spec.dtype,
                device=spec.device,
            ),
        )
    else:
        raise ValueError(f"unsupported KV reservation mode: {config.kv_reservation}")

    if config.num_speculative_tokens:
        if config.speculative_proposer == "chain":
            proposer = NGramChainProposer(
                min_match_length=config.speculative_ngram_min,
                max_match_length=config.speculative_ngram_max,
            )
        else:
            proposer = NGramTrieProposer(
                min_match_length=config.speculative_ngram_min,
                max_match_length=config.speculative_ngram_max,
                max_depth=config.speculative_max_depth,
                max_branching=config.speculative_max_branching,
            )
        decode_handler = SpeculativeDecodeHandler(
            proposer,
            GreedySampler(),
            GreedyTreeAcceptanceSampler(),
            performance_observer,
        )
        decoding_budget = DecodingBudget(
            num_lookahead_tokens=config.num_speculative_tokens,
            max_output_tokens=config.num_speculative_tokens + 1,
        )
    else:
        decode_handler = StandardDecodeHandler(GreedySampler())
        decoding_budget = None

    worker = LocalModelWorker(runner, step_factory, decode_handler)
    model_executor = LocalModelExecutor(
        worker,
        timer=_create_execution_timer(spec),
    )
    scheduler = TokenBudgetScheduler(
        logical_cache,
        max_num_sequences=config.max_num_sequences,
        max_num_scheduled_tokens=config.max_num_scheduled_tokens,
        decoding_budget=decoding_budget,
        short_request_policy=config.short_request_policy,
        self_resubmit_policy=(
            SelfResubmitPolicy(
                max_resubmits=config.max_self_resubmits,
                strict_fallback_rolled_back_tokens=(
                    config.self_resubmit_strict_fallback_rolled_back_tokens
                ),
                initial_extra_blocks=config.self_resubmit_initial_extra_blocks,
                kv_admission_watermark=config.self_resubmit_kv_admission_watermark,
            )
            if config.enable_self_resubmit
            else None
        ),
    )
    ttft_admission = PredictiveTTFTAdmission(
        SlidingWindowStepLatencyPredictor(
            window_size=config.ttft_prediction_window_size,
            min_observations=config.ttft_prediction_min_observations,
            prediction_quantile=config.ttft_prediction_quantile,
        ),
        max_tolerable_ttft_seconds=config.max_tolerable_ttft_seconds,
        max_pending_requests=config.max_pending_requests,
        kv_cache_watermark=config.ttft_kv_cache_watermark,
    )
    engine = EngineCore(
        model_executor,
        scheduler,
        performance_observer=performance_observer,
        ttft_admission=ttft_admission,
        execute_inline=execute_inline,
    )

    def start() -> None:
        runner.load(spec)
        model_executor.initialize()
        # CUDA KV 容量直到 executor 初始化后才确定。
        engine.refresh_performance_metrics()

    return _EngineRuntime(
        engine=engine,
        performance_metrics=performance_observer,
        start=start,
    )


def _create_started_engine_process_runtime(
    spec: ModelSpec,
    config: _EngineRuntimeConfig,
) -> EngineProcessRuntime:
    runtime = _create_engine_runtime(spec, config, execute_inline=True)
    runtime.start()
    return EngineProcessRuntime(
        engine=runtime.engine,
        performance_metrics=runtime.performance_metrics,
        close=runtime.engine.close,
    )


def create_serving_app(
    spec: ModelSpec,
    *,
    runtime: RuntimeMode = "reference",
    kv_reservation: KVReservationMode = "blocks",
    paged_attention_backend: PagedAttentionBackendName = "torch",
    max_num_sequences: int = 8,
    max_num_scheduled_tokens: int = 2048,
    num_kv_blocks: int | None = None,
    kv_block_size: int = 16,
    kv_cache_memory_fraction: float = 0.8,
    enable_prefix_caching: bool = False,
    num_speculative_tokens: int = 0,
    speculative_proposer: SpeculativeProposerName = "chain",
    speculative_ngram_min: int = 2,
    speculative_ngram_max: int = 5,
    speculative_max_depth: int = 4,
    speculative_max_branching: int = 4,
    short_request_policy: ShortRequestPolicy | None = None,
    max_tolerable_ttft_seconds: float | None = None,
    ttft_prediction_window_size: int = 32,
    ttft_prediction_min_observations: int = 100,
    ttft_prediction_quantile: float = 0.9,
    max_pending_requests: int | None = 128,
    ttft_kv_cache_watermark: float | None = 0.9,
    enable_self_resubmit: bool = False,
    max_self_resubmits: int = 2,
    self_resubmit_strict_fallback_rolled_back_tokens: int = 4096,
    self_resubmit_initial_extra_blocks: int = 1,
    self_resubmit_kv_admission_watermark: float = 0.9,
    engine_process: bool = False,
) -> FastAPI:
    """创建 HTTP 服务，并选择 reference 或批量 Engine Core。"""

    # FastAPI 是可选依赖。只有启动 HTTP 服务时才导入，
    # 没有安装它也不影响核心模型功能。
    from light_vllm.serving.http import create_http_app

    close_engine: Callable[[], Awaitable[None]] | None = None
    start_engine: Callable[[], Awaitable[None]] | None = None
    start_runtime: Callable[[], None] | None = None
    performance_metrics: PerformanceMetricsReader | None = None
    if type(num_speculative_tokens) is not int or num_speculative_tokens < 0:
        raise ValueError("num_speculative_tokens must be a non-negative integer")
    if paged_attention_backend not in ("torch", "triton"):
        raise ValueError(f"unsupported paged attention backend: {paged_attention_backend}")
    if speculative_proposer not in ("chain", "trie"):
        raise ValueError(f"unsupported speculative proposer: {speculative_proposer}")
    if runtime == "reference":
        if engine_process:
            raise ValueError("an engine process requires the engine runtime")
        if enable_self_resubmit:
            raise ValueError("self-resubmit requires the engine runtime")
        if max_tolerable_ttft_seconds is not None:
            raise ValueError("TTFT prediction requires the engine runtime")
        if short_request_policy is not None:
            raise ValueError("short-request scheduling requires the engine runtime")
        if num_speculative_tokens:
            raise ValueError("speculative decoding requires the engine runtime")
        if paged_attention_backend != "torch":
            raise ValueError("paged attention backend selection requires the engine runtime")
        # 保留原始单请求基线，继续通过轻量 sync-to-async bridge 对外服务。
        runner = create_runner()
        executor = LocalTokenExecutor(runner, GreedySampler(), device=spec.device)
        service = ReferenceGenerationService(executor)
        engine: EngineClient = InProcessEngineClient(service)
        start_runtime = partial(runner.load, spec)
    else:
        if runtime != "engine":
            raise ValueError(f"unsupported runtime mode: {runtime}")
        if enable_self_resubmit and kv_reservation != "blocks":
            raise ValueError("self-resubmit requires paged KV reservation")
        config = _EngineRuntimeConfig(
            kv_reservation=kv_reservation,
            paged_attention_backend=paged_attention_backend,
            max_num_sequences=max_num_sequences,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            num_kv_blocks=num_kv_blocks,
            kv_block_size=kv_block_size,
            kv_cache_memory_fraction=kv_cache_memory_fraction,
            enable_prefix_caching=enable_prefix_caching,
            num_speculative_tokens=num_speculative_tokens,
            speculative_proposer=speculative_proposer,
            speculative_ngram_min=speculative_ngram_min,
            speculative_ngram_max=speculative_ngram_max,
            speculative_max_depth=speculative_max_depth,
            speculative_max_branching=speculative_max_branching,
            short_request_policy=short_request_policy,
            max_tolerable_ttft_seconds=max_tolerable_ttft_seconds,
            ttft_prediction_window_size=ttft_prediction_window_size,
            ttft_prediction_min_observations=ttft_prediction_min_observations,
            ttft_prediction_quantile=ttft_prediction_quantile,
            max_pending_requests=max_pending_requests,
            ttft_kv_cache_watermark=ttft_kv_cache_watermark,
            enable_self_resubmit=enable_self_resubmit,
            max_self_resubmits=max_self_resubmits,
            self_resubmit_strict_fallback_rolled_back_tokens=(
                self_resubmit_strict_fallback_rolled_back_tokens
            ),
            self_resubmit_initial_extra_blocks=self_resubmit_initial_extra_blocks,
            self_resubmit_kv_admission_watermark=self_resubmit_kv_admission_watermark,
        )
        if engine_process:
            # CUDA 初始化全部发生在 fork 之后，父进程从不持有模型或 KV tensor。
            process_client = ProcessEngineClient(
                partial(_create_started_engine_process_runtime, spec, config),
                start_method="fork",
            )
            engine = process_client
            performance_metrics = process_client
            start_engine = process_client.start
            close_engine = process_client.close
        else:
            engine_runtime = _create_engine_runtime(spec, config)
            engine = engine_runtime.engine
            performance_metrics = engine_runtime.performance_metrics
            start_runtime = engine_runtime.start
            close_engine = engine_runtime.engine.close

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # 模型加载成功后服务才会就绪；加载失败就停止启动。
        if start_runtime is not None:
            start_runtime()
        if start_engine is not None:
            await start_engine()
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


def _optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "off"}:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer or 'off'") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer or 'off'")
    return parsed


def _optional_ratio(value: str) -> float | None:
    if value.lower() in {"none", "off"}:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be within (0, 1] or 'off'") from exc
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be within (0, 1] or 'off'")
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
        "--engine-process",
        action="store_true",
        help="run the stateful Engine and CUDA worker in a dedicated process",
    )
    parser.add_argument(
        "--kv-reservation",
        choices=("blocks", "unbounded"),
        default="blocks",
        help="use physical paged KV or a contiguous unbounded experiment baseline",
    )
    parser.add_argument("--max-num-sequences", type=int, default=8)
    parser.add_argument("--max-num-scheduled-tokens", type=int, default=2048)
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
        help="maximum draft-tree node slots per engine step",
    )
    parser.add_argument(
        "--speculative-proposer",
        choices=("chain", "trie"),
        default="chain",
        help="draft proposer used when speculative decoding is enabled",
    )
    parser.add_argument("--speculative-ngram-min", type=int, default=2)
    parser.add_argument("--speculative-ngram-max", type=int, default=5)
    parser.add_argument(
        "--speculative-max-depth",
        type=int,
        default=4,
        help="maximum depth of an n-gram trie proposal",
    )
    parser.add_argument(
        "--speculative-max-branching",
        type=int,
        default=4,
        help="maximum children retained per n-gram trie node",
    )
    short = parser.add_argument_group("short-request scheduling")
    short.add_argument("--short-request-max-effective-prompt-tokens", type=int)
    short.add_argument("--short-request-max-total-tokens", type=int)
    short.add_argument("--short-request-reserved-scheduled-tokens", type=int)
    short.add_argument("--short-request-reserved-kv-token-slots", type=int)
    short.add_argument("--short-request-reserved-sequences", type=int, default=1)
    short.add_argument("--regular-request-aging-steps", type=int, default=8)
    ttft = parser.add_argument_group("TTFT prediction")
    ttft.add_argument("--max-tolerable-ttft-seconds", type=float)
    ttft.add_argument("--ttft-prediction-window-size", type=int, default=32)
    ttft.add_argument("--ttft-prediction-min-observations", type=int, default=100)
    ttft.add_argument("--ttft-prediction-quantile", type=float, default=0.9)
    ttft.add_argument("--max-pending-requests", type=_optional_positive_int, default=128)
    ttft.add_argument("--ttft-kv-cache-watermark", type=_optional_ratio, default=0.9)
    resubmit = parser.add_argument_group("self-resubmit")
    resubmit.add_argument("--enable-self-resubmit", action="store_true")
    resubmit.add_argument("--max-self-resubmits", type=int, default=2)
    resubmit.add_argument(
        "--self-resubmit-strict-fallback-rolled-back-tokens",
        type=int,
        default=4096,
    )
    resubmit.add_argument(
        "--self-resubmit-initial-extra-blocks",
        type=int,
        default=1,
        help="extra KV blocks guaranteed with the prompt during optimistic admission",
    )
    resubmit.add_argument(
        "--self-resubmit-kv-admission-watermark",
        type=float,
        default=0.9,
        help="fraction of global KV capacity available to new optimistic admissions",
    )
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
            speculative_proposer=args.speculative_proposer,
            speculative_ngram_min=args.speculative_ngram_min,
            speculative_ngram_max=args.speculative_ngram_max,
            speculative_max_depth=args.speculative_max_depth,
            speculative_max_branching=args.speculative_max_branching,
            short_request_policy=_create_short_request_policy(
                max_effective_prompt_tokens=(args.short_request_max_effective_prompt_tokens),
                max_total_tokens=args.short_request_max_total_tokens,
                reserved_scheduled_tokens=(args.short_request_reserved_scheduled_tokens),
                reserved_kv_token_slots=args.short_request_reserved_kv_token_slots,
                reserved_sequences=args.short_request_reserved_sequences,
                regular_aging_steps=args.regular_request_aging_steps,
            ),
            max_tolerable_ttft_seconds=args.max_tolerable_ttft_seconds,
            ttft_prediction_window_size=args.ttft_prediction_window_size,
            ttft_prediction_min_observations=args.ttft_prediction_min_observations,
            ttft_prediction_quantile=args.ttft_prediction_quantile,
            max_pending_requests=args.max_pending_requests,
            ttft_kv_cache_watermark=args.ttft_kv_cache_watermark,
            enable_self_resubmit=args.enable_self_resubmit,
            max_self_resubmits=args.max_self_resubmits,
            self_resubmit_strict_fallback_rolled_back_tokens=(
                args.self_resubmit_strict_fallback_rolled_back_tokens
            ),
            self_resubmit_initial_extra_blocks=(args.self_resubmit_initial_extra_blocks),
            self_resubmit_kv_admission_watermark=(args.self_resubmit_kv_admission_watermark),
            engine_process=args.engine_process,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
