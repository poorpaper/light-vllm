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
from light_vllm.engine.api import EngineClient
from light_vllm.engine.full_sequence import FullSequenceBatchEngine
from light_vllm.engine.in_process import InProcessEngineClient
from light_vllm.execution.local import GreedyFullSequenceBatchExecutor, GreedyTokenExecutor
from light_vllm.generation.reference import ReferenceGenerationService
from light_vllm.models.api import ModelSpec
from light_vllm.scheduler.api import Scheduler
from light_vllm.scheduler.sequence_batching import (
    ContinuousBatchScheduler,
    StaticBatchScheduler,
)

if TYPE_CHECKING:
    from fastapi import FastAPI


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
BatchingMode = Literal["reference", "raw", "continuous"]
# 策略选择只留在 composition root。raw 与 continuous 使用同一个 Engine
# 和执行器，避免调度模式分支扩散到核心热路径。
_SCHEDULER_FACTORIES: dict[str, Callable[[int], Scheduler]] = {
    "raw": lambda max_batch_size: StaticBatchScheduler(max_num_sequences=max_batch_size),
    "continuous": lambda max_batch_size: ContinuousBatchScheduler(max_num_sequences=max_batch_size),
}


def create_serving_app(
    spec: ModelSpec,
    *,
    batching: BatchingMode = "reference",
    max_batch_size: int = 8,
    padding_token_id: int = 0,
) -> FastAPI:
    """创建单进程 HTTP 服务，并选择 reference 或批处理引擎。

    以后换成独立进程引擎时只改这里，不改 HTTP 路由。
    """

    # FastAPI 是可选依赖。只有启动 HTTP 服务时才导入，
    # 没有安装它也不影响核心模型功能。
    from light_vllm.serving.http import create_http_app

    runner = create_runner()
    # reference client 没有常驻 driver；只有批量 Engine 需要在 lifespan
    # 结束时显式等待当前模型迭代完成。
    close_engine: Callable[[], Awaitable[None]] | None = None
    if batching == "reference":
        # 保留原始单请求基线，继续通过轻量 sync-to-async bridge 对外服务。
        executor = GreedyTokenExecutor(runner, device=spec.device)
        service = ReferenceGenerationService(executor)
        engine: EngineClient = InProcessEngineClient(service)
    else:
        # 批量模式只替换 Scheduler，Engine、执行器和 HTTP adapter 完全共用。
        try:
            scheduler_factory = _SCHEDULER_FACTORIES[batching]
        except KeyError as exc:
            raise ValueError(f"unsupported batching mode: {batching}") from exc
        batch_executor = GreedyFullSequenceBatchExecutor(
            runner,
            device=spec.device,
            padding_token_id=padding_token_id,
        )
        batch_engine = FullSequenceBatchEngine(
            batch_executor,
            scheduler_factory(max_batch_size),
        )
        engine = batch_engine
        close_engine = batch_engine.close

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
    parser.add_argument(
        "--batching", choices=("reference", "raw", "continuous"), default="reference"
    )
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--padding-token-id", type=int, default=0)
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
            batching=args.batching,
            max_batch_size=args.max_batch_size,
            padding_token_id=args.padding_token_id,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
