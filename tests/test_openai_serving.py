from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress

import httpx2
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from light_vllm import (
    EngineCapabilities,
    GenerateRequest,
    GenerationError,
    GenerationEvent,
    GenerationFinished,
    GenerationNotReadyError,
    GenerationOverloadedError,
    GenerationRejectedError,
    TokenGenerated,
)
from light_vllm.serving.http import create_http_app
from light_vllm.serving.interfaces import ChatMessage
from light_vllm.serving.openai import OpenAIServingConfig


class PieceDecoder:
    def __init__(self, pieces: dict[int, str]) -> None:
        self._pieces = pieces

    def push(self, token_id: int) -> str:
        return self._pieces[token_id]

    def finish(self) -> str:
        return ""


class FakeTextProcessor:
    eos_token_id = 99
    vocab_size = 128

    def __init__(self) -> None:
        self.chat_messages: tuple[ChatMessage, ...] | None = None

    def encode_prompt(self, prompt: str) -> tuple[int, ...]:
        return (1, 2) if prompt else ()

    def encode_chat(self, messages: tuple[ChatMessage, ...]) -> tuple[int, ...]:
        self.chat_messages = messages
        return (3, 4, 5)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join({10: "hello", 11: "<END>", 12: "tail"}[item] for item in token_ids)

    def new_decoder(self):
        return PieceDecoder({10: "hello<", 11: "END>", 12: "tail"})


class RecordingEngine:
    ready = True
    capabilities = EngineCapabilities()

    def __init__(self, token_ids: tuple[int, ...] = (10, 11, 12)) -> None:
        self.token_ids = token_ids
        self.requests: list[GenerateRequest] = []
        self.closed = False

    async def stream(self, request: GenerateRequest) -> AsyncIterator[GenerationEvent]:
        self.requests.append(request)
        try:
            for position, token_id in enumerate(self.token_ids):
                yield TokenGenerated(token_id, position)
            yield GenerationFinished("length")
        finally:
            self.closed = True

    async def generate(self, request: GenerateRequest):
        raise AssertionError("OpenAI adapter must use EngineClient.stream")


def _client(engine=None, *, speculative=False, timeout=300.0):
    processor = FakeTextProcessor()
    engine = engine or RecordingEngine()
    app = create_http_app(
        engine,
        openai_config=OpenAIServingConfig(
            text_processor=processor,
            served_model_name="local-qwen",
            request_timeout_seconds=timeout,
            speculative_decoding=speculative,
        ),
    )
    return TestClient(app), engine, processor


def _sse_payloads(response) -> list[object]:
    payloads: list[object] = []
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        payloads.append(data if data == "[DONE]" else json.loads(data))
    return payloads


def test_completion_stop_is_hidden_and_counts_generated_tokens() -> None:
    client, engine, _ = _client()

    response = client.post(
        "/v1/completions",
        json={
            "model": "local-qwen",
            "prompt": "hello",
            "max_tokens": 8,
            "temperature": 0.0,
            "stop": "<END>",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["text"] == "hello"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
    assert engine.closed
    assert engine.requests[0].sampling.temperature == 0.0


def test_chat_stream_uses_openai_chunks_and_optional_usage() -> None:
    client, _, processor = _client(RecordingEngine((10,)))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response)
    assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
    assert payloads[1]["choices"][0]["delta"]["content"] == "hello<"
    assert payloads[-2]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "total_tokens": 4,
    }
    assert payloads[-1] == "[DONE]"
    assert processor.chat_messages == (ChatMessage("user", "hi"),)


def test_streaming_and_non_streaming_completion_text_match() -> None:
    non_streaming_client, _, _ = _client()
    streaming_client, _, _ = _client()

    non_streaming = non_streaming_client.post(
        "/v1/completions",
        json={
            "model": "local-qwen",
            "prompt": "hello",
            "max_tokens": 3,
            "temperature": 0.0,
        },
    )
    streaming = streaming_client.post(
        "/v1/completions",
        json={
            "model": "local-qwen",
            "prompt": "hello",
            "max_tokens": 3,
            "temperature": 0.0,
            "stream": True,
        },
    )

    payloads = [item for item in _sse_payloads(streaming) if isinstance(item, dict)]
    streamed_text = "".join(item["choices"][0]["text"] for item in payloads)
    streamed_finish = payloads[-1]["choices"][0]["finish_reason"]
    non_streaming_choice = non_streaming.json()["choices"][0]

    assert streaming.status_code == non_streaming.status_code == 200
    assert streamed_text == non_streaming_choice["text"]
    assert streamed_finish == non_streaming_choice["finish_reason"]


def test_openai_routes_return_standardized_parameter_errors() -> None:
    client, _, _ = _client(speculative=True)

    unknown = client.post(
        "/v1/completions",
        json={"model": "missing", "prompt": "hi", "temperature": 0.0},
    )
    invalid = client.post(
        "/v1/completions",
        json={"model": "local-qwen", "prompt": "hi", "unexpected": True},
    )
    speculative = client.post(
        "/v1/completions",
        json={"model": "local-qwen", "prompt": "hi", "temperature": 1.0},
    )

    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "model_not_found"
    assert invalid.status_code == 400
    assert invalid.json()["error"]["type"] == "invalid_request_error"
    assert speculative.status_code == 400
    assert speculative.json()["error"]["param"] == "temperature"


def test_request_timeout_closes_the_engine_stream() -> None:
    class SlowEngine(RecordingEngine):
        async def stream(self, request):
            self.requests.append(request)
            try:
                await asyncio.sleep(1)
                yield GenerationFinished("length")
            finally:
                self.closed = True

    engine = SlowEngine()
    client, _, _ = _client(engine, timeout=0.01)

    response = client.post(
        "/v1/completions",
        json={"model": "local-qwen", "prompt": "hi", "temperature": 0.0},
    )

    assert response.status_code == 408
    assert response.json()["error"]["code"] == "request_timeout"
    assert engine.closed


@pytest.mark.parametrize(
    ("failure", "status_code", "code"),
    (
        (GenerationRejectedError("too large"), 400, "capacity"),
        (GenerationOverloadedError("too busy"), 429, "overloaded"),
        (GenerationNotReadyError("loading"), 503, "model_not_ready"),
        (GenerationError("broken"), 500, None),
    ),
)
def test_openai_routes_map_runtime_failures(failure, status_code: int, code: str | None) -> None:
    class FailingEngine(RecordingEngine):
        async def stream(self, request):
            if False:
                yield GenerationFinished("length")
            raise failure

    client, _, _ = _client(FailingEngine())

    response = client.post(
        "/v1/completions",
        json={"model": "local-qwen", "prompt": "hi", "temperature": 0.0},
    )

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == code


def test_openai_stream_disconnect_closes_the_engine_stream() -> None:
    class DisconnectEngine(RecordingEngine):
        def __init__(self) -> None:
            super().__init__()
            self.release = asyncio.Event()

        async def stream(self, request):
            self.requests.append(request)
            try:
                yield TokenGenerated(10, 0)
                await self.release.wait()
                yield GenerationFinished("length")
            finally:
                self.closed = True

    async def disconnect_after_first_body() -> None:
        engine = DisconnectEngine()
        processor = FakeTextProcessor()
        app = create_http_app(
            engine,
            openai_config=OpenAIServingConfig(
                text_processor=processor,
                served_model_name="local-qwen",
            ),
        )
        incoming: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        sent: list[dict[str, object]] = []
        await incoming.put(
            {
                "type": "http.request",
                "body": (b'{"model":"local-qwen","prompt":"hi","temperature":0,"stream":true}'),
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
            "path": "/v1/completions",
            "raw_path": b"/v1/completions",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "state": {},
        }

        app_task = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(app_task, timeout=1.0)
            assert engine.closed
            assert any(message.get("status") == 200 for message in sent)
            assert any(b"data: " in message.get("body", b"") for message in sent)
        finally:
            engine.release.set()
            if not app_task.done():
                app_task.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(app_task, timeout=1.0)

    asyncio.run(disconnect_after_first_body())


def test_openai_python_sdk_calls_completion_and_chat_directly() -> None:
    processor = FakeTextProcessor()
    app = create_http_app(
        RecordingEngine(),
        openai_config=OpenAIServingConfig(
            text_processor=processor,
            served_model_name="local-qwen",
        ),
    )

    async def call_sdk() -> None:
        transport = httpx2.ASGITransport(app=app)
        http_client = httpx2.AsyncClient(transport=transport, base_url="http://test")
        client = AsyncOpenAI(
            base_url="http://test/v1",
            api_key="local-only",
            http_client=http_client,
        )
        try:
            completion = await client.completions.create(
                model="local-qwen",
                prompt="hello",
                max_tokens=3,
                temperature=0,
            )
            chat = await client.chat.completions.create(
                model="local-qwen",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=3,
                temperature=0,
            )
        finally:
            await client.close()

        assert completion.choices[0].text == "hello<END>tail"
        assert completion.usage is not None and completion.usage.total_tokens == 5
        assert chat.choices[0].message.content == "hello<END>tail"
        assert chat.usage is not None and chat.usage.total_tokens == 6

    asyncio.run(call_sdk())
