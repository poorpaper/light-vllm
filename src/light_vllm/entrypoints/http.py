"""创建并启动 HTTP 服务。

这里负责选择具体实现、加载模型、解析命令行参数并启动 Uvicorn，
不放推理和 HTTP 编码逻辑。
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from light_vllm.bootstrap import create_runner
from light_vllm.modeling.models.interfaces import ModelSpec
from light_vllm.modeling.runner import ModelRunner
from light_vllm.modeling.tensor_parallel import TensorParallelContext
from light_vllm.runtime.engine.admission import (
    PredictiveTTFTAdmission,
    SlidingWindowStepLatencyPredictor,
)
from light_vllm.runtime.engine.core import EngineCore
from light_vllm.runtime.engine.in_process import InProcessEngineClient
from light_vllm.runtime.engine.interfaces import EngineClient
from light_vllm.runtime.engine.process import EngineProcessRuntime, ProcessEngineClient
from light_vllm.runtime.execution.distributed import (
    SynchronizedPagedKVCachePlanner,
    TensorParallelModelExecutor,
    TorchDistributedGroup,
    run_tensor_parallel_worker,
)
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
    KVCacheManager,
    PagedKVCacheManager,
    UnboundedKVCacheManager,
)
from light_vllm.runtime.observability.interfaces import PerformanceMetricsReader
from light_vllm.runtime.observability.performance import InMemoryPerformanceObserver
from light_vllm.runtime.sampling import ConfigurableSampler, GreedySampler
from light_vllm.runtime.scheduler.interfaces import (
    DecodingBudget,
    SelfResubmitPolicy,
    ShortRequestPolicy,
)
from light_vllm.runtime.scheduler.token_budget import TokenBudgetScheduler

if TYPE_CHECKING:
    from fastapi import FastAPI

    from light_vllm.serving.interfaces import TextProcessor


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
    shutdown: Callable[[], None]


@dataclass(frozen=True, slots=True)
class _ExecutionComponents:
    runner: ModelRunner
    worker: LocalModelWorker
    logical_cache: KVCacheManager
    performance_observer: InMemoryPerformanceObserver
    decoding_budget: DecodingBudget | None


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


def _create_execution_components(
    spec: ModelSpec,
    config: _EngineRuntimeConfig,
    *,
    tensor_parallel_group: TorchDistributedGroup | None = None,
) -> _ExecutionComponents:
    """装配每个 Rank 共用的 Runner、Worker、KV 和 Decode Handler。"""

    parallel = spec.tensor_parallel
    if tensor_parallel_group is None:
        if parallel is not None and parallel.world_size > 1:
            raise ValueError("a multi-rank model requires a tensor parallel executor")
    elif (
        parallel is None
        or parallel.rank != tensor_parallel_group.rank
        or parallel.world_size != tensor_parallel_group.world_size
        or parallel.collectives is not tensor_parallel_group
    ):
        raise ValueError("model tensor parallel context must match the process group")
    if tensor_parallel_group is not None and config.kv_reservation != "blocks":
        # 分页 KV 不持有请求级 tensor；首版 TP 只开放这条生命周期边界清晰的路径。
        raise ValueError("tensor parallel serving requires paged KV reservation")

    runner = create_runner()
    performance_observer = InMemoryPerformanceObserver(spec.architecture)
    if config.kv_reservation == "blocks":
        cache_planner = _create_paged_cache_planner(
            spec,
            num_blocks=config.num_kv_blocks,
            block_size=config.kv_block_size,
            memory_fraction=config.kv_cache_memory_fraction,
        )
        if tensor_parallel_group is not None:
            cache_planner = SynchronizedPagedKVCachePlanner(
                cache_planner,
                tensor_parallel_group,
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
        decode_handler = StandardDecodeHandler(ConfigurableSampler())
        decoding_budget = None

    worker = LocalModelWorker(runner, step_factory, decode_handler)
    return _ExecutionComponents(
        runner=runner,
        worker=worker,
        logical_cache=logical_cache,
        performance_observer=performance_observer,
        decoding_budget=decoding_budget,
    )


def _create_engine_runtime(
    spec: ModelSpec,
    config: _EngineRuntimeConfig,
    *,
    cooperative_inline: bool = False,
    tensor_parallel_group: TorchDistributedGroup | None = None,
) -> _EngineRuntime:
    """装配一个 Engine；模型与 CUDA 资源留到 ``start`` 再初始化。"""

    components = _create_execution_components(
        spec,
        config,
        tensor_parallel_group=tensor_parallel_group,
    )
    if tensor_parallel_group is None:
        model_executor = LocalModelExecutor(
            components.worker,
            timer=_create_execution_timer(spec),
        )

        def shutdown() -> None:
            return None

    else:
        model_executor = TensorParallelModelExecutor(
            LocalModelExecutor(components.worker),
            tensor_parallel_group,
            command_channel=tensor_parallel_group.command_channel,
            timer=_create_execution_timer(spec),
        )
        shutdown = model_executor.shutdown
    scheduler = TokenBudgetScheduler(
        components.logical_cache,
        max_num_sequences=config.max_num_sequences,
        max_num_scheduled_tokens=config.max_num_scheduled_tokens,
        decoding_budget=components.decoding_budget,
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
        performance_observer=components.performance_observer,
        ttft_admission=ttft_admission,
        cooperative_inline=cooperative_inline,
    )

    def start() -> None:
        components.runner.load(spec)
        model_executor.initialize()
        # CUDA KV 容量直到 executor 初始化后才确定。
        engine.refresh_performance_metrics()

    return _EngineRuntime(
        engine=engine,
        performance_metrics=components.performance_observer,
        start=start,
        shutdown=shutdown,
    )


async def _close_engine_runtime(runtime: _EngineRuntime) -> None:
    try:
        await runtime.engine.close()
    finally:
        runtime.shutdown()


def _run_tensor_parallel_rank(
    spec: ModelSpec,
    config: _EngineRuntimeConfig,
    group: TorchDistributedGroup,
) -> None:
    """加载非零 Rank 的本地模型，然后进入统一 Worker 命令循环。"""

    components = _create_execution_components(
        spec,
        config,
        tensor_parallel_group=group,
    )
    components.runner.load(spec)
    run_tensor_parallel_worker(
        LocalModelExecutor(components.worker),
        group,
        command_channel=group.command_channel,
    )


def _create_started_engine_process_runtime(
    spec: ModelSpec,
    config: _EngineRuntimeConfig,
) -> EngineProcessRuntime:
    runtime = _create_engine_runtime(spec, config, cooperative_inline=True)
    runtime.start()
    return EngineProcessRuntime(
        engine=runtime.engine,
        performance_metrics=runtime.performance_metrics,
        close=partial(_close_engine_runtime, runtime),
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
    text_processor: TextProcessor | None = None,
    served_model_name: str | None = None,
    request_timeout_seconds: float | None = 300.0,
    tensor_parallel_group: TorchDistributedGroup | None = None,
) -> FastAPI:
    """创建 HTTP 服务，并选择 reference 或批量 Engine Core。"""

    # FastAPI 是可选依赖。只有启动 HTTP 服务时才导入，
    # 没有安装它也不影响核心模型功能。
    from light_vllm.serving.http import create_http_app
    from light_vllm.serving.openai import OpenAIServingConfig

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
    if (text_processor is None) != (served_model_name is None):
        raise ValueError("text_processor and served_model_name must be configured together")
    if tensor_parallel_group is not None:
        if tensor_parallel_group.rank != 0:
            raise ValueError("only tensor parallel rank 0 can create the HTTP app")
        if runtime != "engine":
            raise ValueError("tensor parallel serving requires the engine runtime")
        if engine_process:
            raise ValueError("tensor parallel ranks already provide the engine process boundary")
    elif spec.tensor_parallel is not None and spec.tensor_parallel.world_size > 1:
        raise ValueError("a multi-rank model requires a tensor parallel process group")
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
        executor = LocalTokenExecutor(runner, ConfigurableSampler(), device=spec.device)
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
            engine_runtime = _create_engine_runtime(
                spec,
                config,
                tensor_parallel_group=tensor_parallel_group,
            )
            engine = engine_runtime.engine
            performance_metrics = engine_runtime.performance_metrics
            start_runtime = engine_runtime.start
            close_engine = partial(_close_engine_runtime, engine_runtime)

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

    openai_config = (
        OpenAIServingConfig(
            text_processor=text_processor,
            served_model_name=served_model_name,
            request_timeout_seconds=request_timeout_seconds,
            speculative_decoding=bool(num_speculative_tokens),
        )
        if text_processor is not None and served_model_name is not None
        else None
    )
    return create_http_app(
        engine,
        lifespan=lifespan,
        performance_metrics=performance_metrics,
        openai_config=openai_config,
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


def _optional_positive_float(value: str) -> float | None:
    if value.lower() in {"none", "off"}:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be positive or 'off'") from exc
    if not isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive or 'off'")
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
    parser.add_argument(
        "--quantization",
        choices=("auto", "none", "awq"),
        default="auto",
        help="weight format; auto reads quantization_config from the checkpoint",
    )
    parser.add_argument(
        "--quantization-backend",
        choices=("auto", "torch", "cuda"),
        default="auto",
        help="AWQ kernel selected once while the model is loaded",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="float32")
    parser.add_argument("--model-args", type=_json_object, default={})
    openai = parser.add_argument_group("local OpenAI-compatible text API")
    openai.add_argument(
        "--tokenizer",
        type=Path,
        help="local tokenizer directory; enables /v1/completions and /v1/chat/completions",
    )
    openai.add_argument(
        "--served-model-name",
        help="model identifier accepted and returned by the OpenAI-compatible endpoints",
    )
    openai.add_argument(
        "--request-timeout-seconds",
        type=_optional_positive_float,
        default=300.0,
        help="whole-request generation timeout, or 'off'",
    )
    parser.add_argument("--runtime", choices=("reference", "engine"), default="reference")
    parser.add_argument(
        "--engine-process",
        action="store_true",
        help="run the stateful Engine and CUDA worker in a dedicated process",
    )
    parallel = parser.add_argument_group("single-node tensor parallelism")
    parallel.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="number of torchrun ranks used to shard one model",
    )
    parallel.add_argument(
        "--distributed-backend",
        choices=("nccl", "gloo"),
        default="nccl",
        help="device collective backend; production GPU serving uses NCCL",
    )
    parallel.add_argument(
        "--distributed-timeout-seconds",
        type=float,
        default=120.0,
        help="maximum wait for a failed or stalled distributed operation",
    )
    parallel.add_argument(
        "--distributed-control-transport",
        choices=("auto", "socket", "gloo"),
        default="auto",
        help="single-node command transport; auto prefers Unix sockets",
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


def _engine_config_from_args(args: argparse.Namespace) -> _EngineRuntimeConfig:
    return _EngineRuntimeConfig(
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
            max_effective_prompt_tokens=args.short_request_max_effective_prompt_tokens,
            max_total_tokens=args.short_request_max_total_tokens,
            reserved_scheduled_tokens=args.short_request_reserved_scheduled_tokens,
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
        self_resubmit_initial_extra_blocks=args.self_resubmit_initial_extra_blocks,
        self_resubmit_kv_admission_watermark=args.self_resubmit_kv_admission_watermark,
    )


def _initialize_tensor_parallel(
    args: argparse.Namespace,
) -> TorchDistributedGroup | None:
    size = args.tensor_parallel_size
    if type(size) is not int or size <= 0:
        raise ValueError("--tensor-parallel-size must be a positive integer")
    torchrun_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if size == 1:
        if torchrun_world_size != 1:
            raise ValueError("torchrun WORLD_SIZE requires matching --tensor-parallel-size")
        return None
    if args.runtime != "engine":
        raise ValueError("tensor parallel serving requires --runtime engine")
    if args.engine_process:
        raise ValueError("tensor parallel serving cannot add a second --engine-process")
    if torchrun_world_size != size:
        raise ValueError("torchrun WORLD_SIZE must equal --tensor-parallel-size")
    if args.loader != "safetensors":
        raise ValueError("tensor parallel serving currently requires --loader safetensors")
    if args.distributed_backend == "nccl" and torch.device(args.device).type != "cuda":
        raise ValueError("NCCL tensor parallelism requires --device cuda")
    if args.distributed_backend == "gloo" and torch.device(args.device).type != "cpu":
        raise ValueError("Gloo tensor parallelism requires --device cpu")
    return TorchDistributedGroup.initialize(
        backend=args.distributed_backend,
        control_transport=args.distributed_control_transport,
        timeout_seconds=args.distributed_timeout_seconds,
    )


def _run_http_entrypoint(
    args: argparse.Namespace,
    group: TorchDistributedGroup | None,
) -> None:
    import uvicorn

    device = group.device if group is not None else torch.device(args.device)
    tensor_parallel = (
        TensorParallelContext(
            rank=group.rank,
            world_size=group.world_size,
            collectives=group,
        )
        if group is not None
        else None
    )
    spec = ModelSpec(
        architecture=args.architecture,
        loader=args.loader,
        model_args=args.model_args,
        weights=args.weights,
        device=device,
        dtype=_DTYPES[args.dtype],
        tensor_parallel=tensor_parallel,
        quantization=args.quantization,
        quantization_backend=args.quantization_backend,
    )
    config = _engine_config_from_args(args)

    if group is not None and group.rank != 0:
        _run_tensor_parallel_rank(spec, config, group)
        return

    text_processor = None
    served_model_name = None
    if args.tokenizer is not None:
        from light_vllm.serving.text import HuggingFaceTextProcessor

        text_processor = HuggingFaceTextProcessor.from_pretrained(args.tokenizer)
        served_model_name = args.served_model_name or (
            args.weights.name if args.weights is not None else args.architecture
        )
    elif args.served_model_name is not None:
        raise ValueError("--served-model-name requires --tokenizer")

    uvicorn.run(
        create_serving_app(
            spec,
            runtime=args.runtime,
            kv_reservation=config.kv_reservation,
            paged_attention_backend=config.paged_attention_backend,
            max_num_sequences=config.max_num_sequences,
            max_num_scheduled_tokens=config.max_num_scheduled_tokens,
            num_kv_blocks=config.num_kv_blocks,
            kv_block_size=config.kv_block_size,
            kv_cache_memory_fraction=config.kv_cache_memory_fraction,
            enable_prefix_caching=config.enable_prefix_caching,
            num_speculative_tokens=config.num_speculative_tokens,
            speculative_proposer=config.speculative_proposer,
            speculative_ngram_min=config.speculative_ngram_min,
            speculative_ngram_max=config.speculative_ngram_max,
            speculative_max_depth=config.speculative_max_depth,
            speculative_max_branching=config.speculative_max_branching,
            short_request_policy=config.short_request_policy,
            max_tolerable_ttft_seconds=config.max_tolerable_ttft_seconds,
            ttft_prediction_window_size=config.ttft_prediction_window_size,
            ttft_prediction_min_observations=config.ttft_prediction_min_observations,
            ttft_prediction_quantile=config.ttft_prediction_quantile,
            max_pending_requests=config.max_pending_requests,
            ttft_kv_cache_watermark=config.ttft_kv_cache_watermark,
            enable_self_resubmit=config.enable_self_resubmit,
            max_self_resubmits=config.max_self_resubmits,
            self_resubmit_strict_fallback_rolled_back_tokens=(
                config.self_resubmit_strict_fallback_rolled_back_tokens
            ),
            self_resubmit_initial_extra_blocks=config.self_resubmit_initial_extra_blocks,
            self_resubmit_kv_admission_watermark=config.self_resubmit_kv_admission_watermark,
            engine_process=args.engine_process,
            text_processor=text_processor,
            served_model_name=served_model_name,
            request_timeout_seconds=args.request_timeout_seconds,
            tensor_parallel_group=group,
        ),
        host=args.host,
        port=args.port,
    )


def main() -> None:
    args = _create_parser().parse_args()
    group = _initialize_tensor_parallel(args)
    try:
        _run_http_entrypoint(args, group)
    finally:
        if group is not None:
            group.close()


if __name__ == "__main__":
    main()
