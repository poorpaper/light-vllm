import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress

import pytest
from fastapi.testclient import TestClient

from light_vllm import (
    EngineCapabilities,
    GenerateRequest,
    GenerateResult,
    GenerationError,
    GenerationEvent,
    GenerationFinished,
    GenerationNotReadyError,
    GenerationOverloadedError,
    ModelSpec,
    TokenGenerated,
)
from light_vllm.entrypoints.http import create_serving_app
from light_vllm.runtime.scheduler import ShortRequestPolicy
from light_vllm.serving.http import _encoded_stream, create_http_app


class StubEngineClient:
    ready = True
    capabilities = EngineCapabilities()

    async def stream(self, request: GenerateRequest) -> AsyncIterator[GenerationEvent]:
        yield TokenGenerated(token_id=7, position=0)
        yield GenerationFinished(finish_reason="length")

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        return GenerateResult(
            input_ids=request.input_ids,
            generated_token_ids=(7,),
            finish_reason="length",
        )


class UnreadyEngineClient(StubEngineClient):
    ready = False

    async def stream(self, request: GenerateRequest) -> AsyncIterator[GenerationEvent]:
        raise GenerationNotReadyError("not ready")
        yield

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        raise GenerationNotReadyError("not ready")


class FailingStreamEngineClient(StubEngineClient):
    async def stream(self, request: GenerateRequest) -> AsyncIterator[GenerationEvent]:
        yield TokenGenerated(token_id=7, position=0)
        raise GenerationError("failed after the response started")


class OverloadedEngineClient(StubEngineClient):
    async def stream(self, request: GenerateRequest) -> AsyncIterator[GenerationEvent]:
        raise GenerationOverloadedError("predicted TTFT exceeds SLO")
        yield

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        raise GenerationOverloadedError("predicted TTFT exceeds SLO")


class CloseTrackingAsyncIterator(AsyncIterator[GenerationEvent]):
    def __init__(self) -> None:
        self.closed = False

    async def __anext__(self) -> GenerationEvent:
        return GenerationFinished(finish_reason="length")

    async def aclose(self) -> None:
        self.closed = True


class DisconnectTrackingEngineClient(StubEngineClient):
    def __init__(self) -> None:
        self.closed = False
        self.release = asyncio.Event()

    async def stream(self, request: GenerateRequest) -> AsyncIterator[GenerationEvent]:
        try:
            yield TokenGenerated(token_id=7, position=0)
            await self.release.wait()
        finally:
            self.closed = True


class FirstEventFailureIterator(AsyncIterator[GenerationEvent]):
    def __init__(self) -> None:
        self.closed = False

    async def __anext__(self) -> GenerationEvent:
        raise GenerationError("failed before the response started")

    async def aclose(self) -> None:
        self.closed = True


class FirstEventFailureEngineClient(StubEngineClient):
    def __init__(self) -> None:
        self.events = FirstEventFailureIterator()

    def stream(self, request: GenerateRequest) -> AsyncIterator[GenerationEvent]:
        return self.events


def test_http_adapter_exposes_health_and_non_streaming_generation() -> None:
    client = TestClient(create_http_app(StubEngineClient()))

    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}
    assert client.get("/capabilities").json()["max_request_tokens"] is None

    response = client.post("/generate", json={"input_ids": [1, 2], "max_new_tokens": 1})

    assert response.status_code == 200
    assert response.json() == {
        "input_ids": [1, 2],
        "generated_token_ids": [7],
        "token_ids": [1, 2, 7],
        "finish_reason": "length",
    }


def test_http_adapter_streams_generation_events_as_sse() -> None:
    client = TestClient(create_http_app(StubEngineClient()))

    response = client.post("/generate/stream", json={"input_ids": [1], "max_new_tokens": 1})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: token" in response.text
    assert 'data: {"token_id":7,"position":0}' in response.text
    assert "event: done" in response.text
    assert 'data: {"finish_reason":"length"}' in response.text


def test_stream_failure_after_first_event_is_encoded_in_the_stream() -> None:
    client = TestClient(create_http_app(FailingStreamEngineClient()))

    response = client.post("/generate/stream", json={"input_ids": [1]})

    assert response.status_code == 200
    assert "event: token" in response.text
    assert "event: error" in response.text
    assert '"code":"generation_failed"' in response.text


def test_sse_bridge_closes_the_core_stream_when_cancelled() -> None:
    events = CloseTrackingAsyncIterator()

    async def cancel_after_first_event() -> None:
        stream = _encoded_stream(TokenGenerated(token_id=7, position=0), events)
        assert b"event: token" in await anext(stream)
        await stream.aclose()

    asyncio.run(cancel_after_first_event())

    assert events.closed


def test_asgi_disconnect_closes_the_engine_stream() -> None:
    async def disconnect_after_first_body() -> None:
        engine = DisconnectTrackingEngineClient()
        app = create_http_app(engine)
        incoming: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        sent: list[dict[str, object]] = []
        await incoming.put(
            {
                "type": "http.request",
                "body": b'{"input_ids":[1],"max_new_tokens":2}',
                "more_body": False,
            }
        )

        async def receive() -> dict[str, object]:
            return await incoming.get()

        async def send(message: dict[str, object]) -> None:
            sent.append(message)
            if message["type"] == "http.response.body" and message.get("body"):
                await incoming.put({"type": "http.disconnect"})

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/generate/stream",
            "raw_path": b"/generate/stream",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "state": {},
        }

        app_task = asyncio.create_task(app(scope, receive, send))
        try:
            done, _ = await asyncio.wait({app_task}, timeout=1.0)
            assert app_task in done
            await app_task

            assert engine.closed
            assert any(message.get("status") == 200 for message in sent)
            assert any(b"event: token" in message.get("body", b"") for message in sent)
        finally:
            # 测试失败时也要放行，避免 asyncio.run() 一直等这个故意阻塞的流。
            engine.release.set()
            if not app_task.done():
                app_task.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(app_task, timeout=1.0)

    asyncio.run(disconnect_after_first_body())


def test_not_ready_is_mapped_before_a_response_or_stream_starts() -> None:
    client = TestClient(create_http_app(UnreadyEngineClient()))

    assert client.get("/readyz").status_code == 503
    assert client.post("/generate", json={"input_ids": [1]}).status_code == 503
    assert client.post("/generate/stream", json={"input_ids": [1]}).status_code == 503


def test_predicted_ttft_overload_is_mapped_to_429_before_streaming() -> None:
    client = TestClient(create_http_app(OverloadedEngineClient()))

    generated = client.post("/generate", json={"input_ids": [1]})
    streamed = client.post("/generate/stream", json={"input_ids": [1]})

    assert generated.status_code == 429
    assert streamed.status_code == 429
    assert generated.json()["detail"] == "predicted TTFT exceeds SLO"


def test_stream_is_closed_when_the_first_event_fails() -> None:
    engine = FirstEventFailureEngineClient()
    client = TestClient(create_http_app(engine))

    response = client.post("/generate/stream", json={"input_ids": [1]})

    assert response.status_code == 500
    assert engine.events.closed


def test_http_request_validation_stays_in_the_adapter() -> None:
    client = TestClient(create_http_app(StubEngineClient()))

    response = client.post(
        "/generate",
        json={"input_ids": [], "max_new_tokens": 0, "unexpected": True},
    )

    assert response.status_code == 422


def test_engine_capabilities_replace_transport_token_limits() -> None:
    app = create_serving_app(
        ModelSpec(
            architecture="tiny-attention-causal-lm",
            model_args={"vocab_size": 16, "hidden_size": 4, "num_heads": 1},
        ),
        runtime="engine",
        num_kv_blocks=2,
        kv_block_size=2,
    )

    with TestClient(app) as client:
        capabilities = client.get("/capabilities")
        rejected = client.post(
            "/generate",
            json={"input_ids": [1, 2, 3], "max_new_tokens": 2},
        )

    assert capabilities.json()["max_request_tokens"] == 4
    assert rejected.status_code == 422
    assert "engine supports at most 4" in rejected.json()["detail"]


def test_tiny_model_serves_an_end_to_end_http_request() -> None:
    app = create_serving_app(
        ModelSpec(
            architecture="tiny-causal-lm",
            model_args={"vocab_size": 16, "hidden_size": 4},
        )
    )

    with TestClient(app) as client:
        assert client.get("/readyz").status_code == 200
        response = client.post(
            "/generate",
            json={"input_ids": [1, 2], "max_new_tokens": 2},
        )

    assert response.status_code == 200
    assert response.json()["input_ids"] == [1, 2]
    assert len(response.json()["generated_token_ids"]) == 2
    assert len(response.json()["token_ids"]) == 4


def test_tiny_attention_model_serves_through_engine_core() -> None:
    app = create_serving_app(
        ModelSpec(
            architecture="tiny-attention-causal-lm",
            model_args={"vocab_size": 16, "hidden_size": 4, "num_heads": 1},
        ),
        runtime="engine",
        max_num_sequences=2,
        max_num_scheduled_tokens=2,
        num_kv_blocks=256,
    )

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={"input_ids": [1, 2], "max_new_tokens": 2},
        )

    assert response.status_code == 200
    assert len(response.json()["generated_token_ids"]) == 2


def test_engine_runtime_composes_ngram_trie_speculation() -> None:
    app = create_serving_app(
        ModelSpec(
            architecture="tiny-attention-causal-lm",
            model_args={"vocab_size": 16, "hidden_size": 4, "num_heads": 1},
        ),
        runtime="engine",
        max_num_sequences=1,
        max_num_scheduled_tokens=8,
        num_kv_blocks=256,
        num_speculative_tokens=2,
        speculative_proposer="trie",
        speculative_ngram_min=2,
        speculative_ngram_max=2,
        speculative_max_depth=2,
        speculative_max_branching=2,
    )

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={"input_ids": [1, 2, 3, 1, 2], "max_new_tokens": 2},
        )
        metrics = client.get("/metrics")

    assert response.status_code == 200
    assert len(response.json()["generated_token_ids"]) == 2
    assert "light_vllm_speculation_attempts_total" in metrics.text
    assert "light_vllm_speculative_verified_tokens_total" in metrics.text


def test_engine_runtime_rejects_an_unknown_speculative_proposer() -> None:
    with pytest.raises(ValueError, match="unsupported speculative proposer"):
        create_serving_app(
            ModelSpec(
                architecture="tiny-attention-causal-lm",
                model_args={"vocab_size": 16, "hidden_size": 4, "num_heads": 1},
            ),
            runtime="engine",
            num_kv_blocks=4,
            speculative_proposer="unknown",
        )


def test_engine_runtime_accepts_short_request_policy() -> None:
    app = create_serving_app(
        ModelSpec(
            architecture="tiny-attention-causal-lm",
            model_args={"vocab_size": 16, "hidden_size": 4, "num_heads": 1},
        ),
        runtime="engine",
        max_num_sequences=2,
        max_num_scheduled_tokens=4,
        num_kv_blocks=32,
        short_request_policy=ShortRequestPolicy(
            max_effective_prompt_tokens=2,
            max_total_tokens=4,
            reserved_scheduled_tokens=2,
            reserved_kv_token_slots=3,
        ),
        enable_self_resubmit=True,
        max_self_resubmits=1,
        self_resubmit_strict_fallback_rolled_back_tokens=8,
    )

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={"input_ids": [1, 2], "max_new_tokens": 2},
        )

    assert response.status_code == 200
    assert len(response.json()["generated_token_ids"]) == 2


def test_self_resubmit_rejects_an_unbounded_kv_backend() -> None:
    with pytest.raises(ValueError, match="requires paged KV"):
        create_serving_app(
            ModelSpec(
                architecture="tiny-attention-causal-lm",
                model_args={"vocab_size": 16, "hidden_size": 4, "num_heads": 1},
            ),
            runtime="engine",
            kv_reservation="unbounded",
            enable_self_resubmit=True,
        )


@pytest.mark.parametrize("architecture", ("qwen2", "qwen2.5"))
def test_qwen_engine_runtime_exposes_model_agnostic_performance_metrics(
    architecture: str,
) -> None:
    app = create_serving_app(
        ModelSpec(
            architecture=architecture,
            model_args={
                "model_type": "qwen2",
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "max_position_embeddings": 32,
                "tie_word_embeddings": True,
            },
        ),
        runtime="engine",
        max_num_sequences=2,
        max_num_scheduled_tokens=4,
        num_kv_blocks=16,
        kv_block_size=2,
    )

    with TestClient(app) as client:
        generated = client.post(
            "/generate",
            json={"input_ids": [1, 2], "max_new_tokens": 2},
        )
        metrics = client.get("/metrics")

    assert generated.status_code == 200
    assert metrics.status_code == 200
    assert (
        f'light_vllm_time_to_first_token_seconds_count{{model="{architecture}"}} 1' in metrics.text
    )
    assert (
        f'light_vllm_inter_token_latency_seconds_count{{model="{architecture}"}} 1' in metrics.text
    )
    assert f'light_vllm_engine_step_seconds_count{{model="{architecture}"' in metrics.text
    assert (
        f'light_vllm_requests_total{{model="{architecture}",outcome="finished"}} 1' in metrics.text
    )


def test_engine_core_serves_with_unbounded_kv_reservations() -> None:
    app = create_serving_app(
        ModelSpec(
            architecture="tiny-attention-causal-lm",
            model_args={"vocab_size": 16, "hidden_size": 4, "num_heads": 1},
        ),
        runtime="engine",
        kv_reservation="unbounded",
        max_num_sequences=2,
        max_num_scheduled_tokens=2,
    )

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={"input_ids": [1, 2], "max_new_tokens": 2},
        )

    assert response.status_code == 200
    assert len(response.json()["generated_token_ids"]) == 2
