from __future__ import annotations

import asyncio
import multiprocessing
from threading import Thread

import pytest

from light_vllm.runtime.engine.interfaces import EngineCapabilities
from light_vllm.runtime.engine.process import (
    ConnectionEngineClient,
    EngineProcessRuntime,
    ProcessEngineClient,
    run_engine_server,
)
from light_vllm.runtime.generation.interfaces import (
    GenerateRequest,
    GenerateResult,
    GenerationFinished,
    GenerationOverloadedError,
    TokenGenerated,
)
from light_vllm.runtime.observability.performance import InMemoryPerformanceObserver


class _FakeProcessEngine:
    @property
    def ready(self) -> bool:
        return True

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(max_model_tokens=128, max_num_sequences=4)

    async def stream(self, request: GenerateRequest):
        if request.input_ids == (99,):
            raise GenerationOverloadedError("fake overload")
        for position in range(request.max_new_tokens):
            await asyncio.sleep(0)
            yield TokenGenerated(token_id=request.input_ids[-1] + position + 1, position=position)
        yield GenerationFinished(finish_reason="length")

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        token_ids: list[int] = []
        async for event in self.stream(request):
            if isinstance(event, TokenGenerated):
                token_ids.append(event.token_id)
        return GenerateResult(request.input_ids, tuple(token_ids), "length")

    async def close(self) -> None:
        return None


def _create_fake_process_runtime() -> EngineProcessRuntime:
    engine = _FakeProcessEngine()
    metrics = InMemoryPerformanceObserver("fake-process-model")
    return EngineProcessRuntime(engine=engine, performance_metrics=metrics, close=engine.close)


def test_process_engine_client_streams_metrics_errors_and_closes() -> None:
    async def run() -> None:
        client = ProcessEngineClient(
            _create_fake_process_runtime,
            start_method="spawn",
            startup_timeout_seconds=30,
        )
        await client.start()
        assert client.ready
        assert client.capabilities.max_model_tokens == 128

        result = await client.generate(GenerateRequest(input_ids=(4,), max_new_tokens=3))
        assert result.generated_token_ids == (5, 6, 7)

        snapshot = await asyncio.to_thread(client.snapshot)
        assert snapshot.model_name == "fake-process-model"

        events = client.stream(GenerateRequest(input_ids=(99,), max_new_tokens=1))
        with pytest.raises(GenerationOverloadedError, match="fake overload"):
            await anext(events)
        await events.aclose()

        await client.close()
        assert not client.ready

    asyncio.run(run())


def test_connection_engine_client_uses_an_external_engine_owner() -> None:
    async def run() -> None:
        client_connection, engine_connection = multiprocessing.Pipe(duplex=True)
        server = Thread(
            target=run_engine_server,
            args=(engine_connection, _create_fake_process_runtime),
            daemon=True,
        )
        server.start()
        client = ConnectionEngineClient(
            client_connection,
            startup_timeout_seconds=5,
            shutdown_timeout_seconds=5,
        )

        await client.start()
        assert client.ready
        assert client.capabilities.max_model_tokens == 128
        results = await asyncio.gather(
            *(
                client.generate(GenerateRequest(input_ids=(token,), max_new_tokens=2))
                for token in range(7, 11)
            )
        )
        assert [result.generated_token_ids for result in results] == [
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
        ]
        assert (await asyncio.to_thread(client.snapshot)).model_name == "fake-process-model"

        await client.close()
        await asyncio.to_thread(server.join, 5)
        assert not server.is_alive()
        assert not client.ready

    asyncio.run(run())
