"""Executable composition root for the HTTP server.

Unlike ``serving.http``, this module is allowed to choose concrete runtime
implementations, manage model startup, parse CLI configuration, and invoke
Uvicorn. No inference or HTTP encoding logic should be implemented here.
"""

from __future__ import annotations

import argparse
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from light_vllm.bootstrap import create_runner
from light_vllm.contracts import ModelSpec
from light_vllm.engine import InProcessEngineClient
from light_vllm.generation import GreedyGenerationService

if TYPE_CHECKING:
    from fastapi import FastAPI


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def create_serving_app(spec: ModelSpec) -> FastAPI:
    """Wire the current single-process reference stack into the HTTP adapter.

    Replacing ``InProcessEngineClient`` with a future process client belongs in
    this composition root; ``serving.http`` should remain unchanged.
    """

    # FastAPI is an optional serving dependency. Importing it lazily keeps the
    # model/runtime package usable when only the core dependencies are installed.
    from light_vllm.serving.http import create_http_app

    runner = create_runner()
    service = GreedyGenerationService(runner, device=spec.device)
    engine = InProcessEngineClient(service)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # The application does not become ready until the candidate model has
        # loaded successfully. A failed load aborts startup instead of exposing
        # a partially initialized engine.
        runner.load(spec)
        yield

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
    uvicorn.run(create_serving_app(spec), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
