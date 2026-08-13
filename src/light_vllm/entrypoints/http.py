"""创建并启动 HTTP 服务。

这里负责选择具体实现、加载模型、解析命令行参数并启动 Uvicorn，
不放推理和 HTTP 编码逻辑。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from light_vllm.bootstrap import create_runner
from light_vllm.modeling.models.interfaces import ModelSpec
from light_vllm.runtime.engine.core import EngineCore
from light_vllm.runtime.engine.in_process import InProcessEngineClient
from light_vllm.runtime.engine.interfaces import EngineClient
from light_vllm.runtime.execution.local import LocalModelExecutor, LocalTokenExecutor
from light_vllm.runtime.generation.reference import ReferenceGenerationService
from light_vllm.runtime.kv_cache import (
    ContiguousKVCache,
    KVCacheManager,
    KVCacheSpec,
    PagedKVCacheManager,
    UnboundedKVCacheManager,
)
from light_vllm.runtime.sampling import GreedySampler
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


def _create_kv_manager(
    mode: KVReservationMode,
    *,
    num_blocks: int,
    block_size: int,
) -> KVCacheManager:
    if mode == "blocks":
        return PagedKVCacheManager(num_blocks=num_blocks, block_size=block_size)
    if mode == "unbounded":
        return UnboundedKVCacheManager()
    raise ValueError(f"unsupported KV reservation mode: {mode}")


def create_serving_app(
    spec: ModelSpec,
    *,
    runtime: RuntimeMode = "reference",
    kv_reservation: KVReservationMode = "blocks",
    max_num_sequences: int = 8,
    max_num_scheduled_tokens: int = 256,
    num_kv_blocks: int = 256,
    kv_block_size: int = 16,
    kv_num_layers: int = 1,
    kv_num_heads: int = 1,
    kv_head_size: int = 64,
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
    sampler = GreedySampler()
    if runtime == "reference":
        # 保留原始单请求基线，继续通过轻量 sync-to-async bridge 对外服务。
        executor = LocalTokenExecutor(runner, sampler, device=spec.device)
        service = ReferenceGenerationService(executor)
        engine: EngineClient = InProcessEngineClient(service)
    else:
        if runtime != "engine":
            raise ValueError(f"unsupported runtime mode: {runtime}")
        tensor_cache = ContiguousKVCache(
            KVCacheSpec(
                num_layers=kv_num_layers,
                num_kv_heads=kv_num_heads,
                head_size=kv_head_size,
                dtype=spec.dtype,
                device=spec.device,
            )
        )
        logical_cache = _create_kv_manager(
            kv_reservation,
            num_blocks=num_kv_blocks,
            block_size=kv_block_size,
        )
        model_executor = LocalModelExecutor(
            runner,
            tensor_cache,
            sampler,
            device=spec.device,
        )
        scheduler = TokenBudgetScheduler(
            logical_cache,
            max_num_sequences=max_num_sequences,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
        )
        engine_core = EngineCore(model_executor, scheduler)
        engine = engine_core
        close_engine = engine_core.close

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # 模型加载成功后服务才会就绪；加载失败就停止启动。
        runner.load(spec)
        try:
            yield
        finally:
            # close 会先拒绝新请求，再等待正在运行的同步模型步骤安全结束。
            if close_engine is not None:
                await close_engine()

    return create_http_app(engine, lifespan=lifespan)


def _json_object(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("model args must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("model args must be a JSON object")
    return parsed


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
        help="use logical KV blocks or an unbounded no-block experiment baseline",
    )
    parser.add_argument("--max-num-sequences", type=int, default=8)
    parser.add_argument("--max-num-scheduled-tokens", type=int, default=256)
    parser.add_argument("--num-kv-blocks", type=int, default=256)
    parser.add_argument("--kv-block-size", type=int, default=16)
    parser.add_argument("--kv-num-layers", type=int, default=1)
    parser.add_argument("--kv-num-heads", type=int, default=1)
    parser.add_argument("--kv-head-size", type=int, default=64)
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
            max_num_sequences=args.max_num_sequences,
            max_num_scheduled_tokens=args.max_num_scheduled_tokens,
            num_kv_blocks=args.num_kv_blocks,
            kv_block_size=args.kv_block_size,
            kv_num_layers=args.kv_num_layers,
            kv_num_heads=args.kv_num_heads,
            kv_head_size=args.kv_head_size,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
