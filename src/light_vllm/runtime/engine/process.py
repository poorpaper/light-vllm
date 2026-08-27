"""把 HTTP 控制面与有状态推理引擎隔离到不同进程。"""

from __future__ import annotations

import asyncio
import multiprocessing
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from itertools import count
from multiprocessing.connection import Connection

from light_vllm.runtime.engine.interfaces import EngineCapabilities, EngineClient
from light_vllm.runtime.generation.interfaces import (
    GenerateRequest,
    GenerateResult,
    GenerationError,
    GenerationEvent,
    GenerationFinished,
    GenerationNotReadyError,
    GenerationOverloadedError,
    GenerationRejectedError,
    TokenGenerated,
)
from light_vllm.runtime.observability.interfaces import (
    PerformanceMetricsReader,
    PerformanceSnapshot,
)


@dataclass(frozen=True, slots=True)
class EngineProcessRuntime:
    """子进程启动完成后的 Engine 与控制面读取端口。"""

    engine: EngineClient
    performance_metrics: PerformanceMetricsReader
    close: Callable[[], Awaitable[None]]


EngineProcessFactory = Callable[[], EngineProcessRuntime]


@dataclass(frozen=True, slots=True)
class _StartStream:
    request_id: int
    request: GenerateRequest


@dataclass(frozen=True, slots=True)
class _CancelStream:
    request_id: int


@dataclass(frozen=True, slots=True)
class _ReadMetrics:
    call_id: int


@dataclass(frozen=True, slots=True)
class _Shutdown:
    pass


_Command = _StartStream | _CancelStream | _ReadMetrics | _Shutdown


@dataclass(frozen=True, slots=True)
class _Started:
    capabilities: EngineCapabilities
    snapshot: PerformanceSnapshot


@dataclass(frozen=True, slots=True)
class _StreamEvent:
    request_id: int
    event: GenerationEvent


@dataclass(frozen=True, slots=True)
class _StreamFailure:
    request_id: int
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class _MetricsResult:
    call_id: int
    snapshot: PerformanceSnapshot


@dataclass(frozen=True, slots=True)
class _ProcessFailure:
    detail: str


@dataclass(frozen=True, slots=True)
class _Stopped:
    pass


_SingleResponse = (
    _Started | _StreamEvent | _StreamFailure | _MetricsResult | _ProcessFailure | _Stopped
)


@dataclass(frozen=True, slots=True)
class _ResponseBatch:
    responses: tuple[_SingleResponse, ...]


_Response = _SingleResponse | _ResponseBatch


@dataclass(slots=True)
class _MetricsWaiter:
    ready: threading.Event
    snapshot: PerformanceSnapshot | None = None
    error: Exception | None = None


def _failure_kind(exc: Exception) -> str:
    if isinstance(exc, GenerationNotReadyError):
        return "not_ready"
    if isinstance(exc, GenerationOverloadedError):
        return "overloaded"
    if isinstance(exc, GenerationRejectedError):
        return "rejected"
    return "generation"


def _remote_error(kind: str, detail: str) -> GenerationError:
    if kind == "not_ready":
        return GenerationNotReadyError(detail)
    if kind == "overloaded":
        return GenerationOverloadedError(detail)
    if kind == "rejected":
        return GenerationRejectedError(detail)
    return GenerationError(detail)


async def _forward_stream(
    responses: asyncio.Queue[_SingleResponse | None],
    engine: EngineClient,
    request_id: int,
    request: GenerateRequest,
) -> None:
    events = engine.stream(request)
    try:
        async for event in events:
            responses.put_nowait(_StreamEvent(request_id, event))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        responses.put_nowait(
            _StreamFailure(
                request_id=request_id,
                kind=_failure_kind(exc),
                detail=str(exc),
            )
        )
    finally:
        close = getattr(events, "aclose", None)
        if close is not None:
            await close()


async def _write_responses(
    connection: Connection,
    responses: asyncio.Queue[_SingleResponse | None],
) -> None:
    """把同一事件循环轮次已就绪的响应合并成一次 IPC 写入。"""

    while True:
        first = await responses.get()
        if first is None:
            return
        ready = [first]
        while not responses.empty():
            response = responses.get_nowait()
            if response is None:
                _send_ready_responses(connection, ready)
                return
            ready.append(response)
        _send_ready_responses(connection, ready)


def _send_ready_responses(
    connection: Connection,
    responses: list[_SingleResponse],
) -> None:
    # 单请求不额外包一层；并发请求才共享一次 pickle、Pipe 写入和前端唤醒。
    connection.send(responses[0] if len(responses) == 1 else _ResponseBatch(tuple(responses)))


def _read_commands(
    connection: Connection,
    loop: asyncio.AbstractEventLoop,
    commands: asyncio.Queue[_Command | None],
) -> None:
    try:
        while True:
            ready = [connection.recv()]
            # 一次唤醒交付 Pipe 中已经 ready 的完整 burst，避免第一个请求
            # 单独启动 Engine；不等待未来命令，也不引入墙钟 batching 窗口。
            while connection.poll():
                ready.append(connection.recv())
            loop.call_soon_threadsafe(_enqueue_commands, commands, tuple(ready))
    except (EOFError, OSError):
        # server 可能已经完成关闭；此时 EOF 只是 reader 的最终清理信号。
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(commands.put_nowait, None)


def _enqueue_commands(
    queue: asyncio.Queue[_Command | None],
    commands: tuple[_Command, ...],
) -> None:
    for command in commands:
        queue.put_nowait(command)


async def serve_engine_connection(
    connection: Connection,
    factory: EngineProcessFactory,
) -> None:
    """在当前进程服务一条 Engine IPC 连接。"""

    runtime = factory()
    streams: dict[int, asyncio.Task[None]] = {}
    responses: asyncio.Queue[_SingleResponse | None] | None = None
    writer: asyncio.Task[None] | None = None
    graceful_shutdown = False
    try:
        if not runtime.engine.ready:
            raise GenerationNotReadyError("engine process did not become ready")

        connection.send(
            _Started(
                capabilities=runtime.engine.capabilities,
                snapshot=runtime.performance_metrics.snapshot(),
            )
        )
        commands: asyncio.Queue[_Command | None] = asyncio.Queue()
        responses = asyncio.Queue()
        writer = asyncio.create_task(
            _write_responses(connection, responses),
            name="light-vllm-engine-ipc-writer",
        )

        def wake_on_writer_failure(task: asyncio.Task[None]) -> None:
            if not task.cancelled() and task.exception() is not None:
                commands.put_nowait(None)

        writer.add_done_callback(wake_on_writer_failure)
        loop = asyncio.get_running_loop()
        reader = threading.Thread(
            target=_read_commands,
            args=(connection, loop, commands),
            name="light-vllm-engine-ipc-reader",
            daemon=True,
        )
        reader.start()
        while True:
            command = await commands.get()
            if command is None:
                if writer.done():
                    await writer
                break
            if isinstance(command, _Shutdown):
                graceful_shutdown = True
                break
            if isinstance(command, _StartStream):
                if command.request_id in streams:
                    raise RuntimeError("duplicate process-engine request ID")
                task = asyncio.create_task(
                    _forward_stream(
                        responses,
                        runtime.engine,
                        command.request_id,
                        command.request,
                    ),
                    name=f"light-vllm-process-stream-{command.request_id}",
                )
                streams[command.request_id] = task
                task.add_done_callback(
                    lambda _task, request_id=command.request_id: streams.pop(request_id, None)
                )
                continue
            if isinstance(command, _CancelStream):
                task = streams.get(command.request_id)
                if task is not None:
                    task.cancel()
                continue
            if isinstance(command, _ReadMetrics):
                responses.put_nowait(
                    _MetricsResult(
                        call_id=command.call_id,
                        snapshot=runtime.performance_metrics.snapshot(),
                    )
                )
                continue
            raise TypeError(f"unsupported process-engine command: {type(command).__name__}")
    finally:
        for task in tuple(streams.values()):
            task.cancel()
        if streams:
            await asyncio.gather(*streams.values(), return_exceptions=True)
        runtime_closed = False
        try:
            await runtime.close()
            runtime_closed = True
        finally:
            if writer is not None and responses is not None:
                if graceful_shutdown and runtime_closed:
                    if not writer.done():
                        responses.put_nowait(_Stopped())
                        responses.put_nowait(None)
                    await writer
                else:
                    writer.cancel()
                    await asyncio.gather(writer, return_exceptions=True)


def run_engine_server(
    connection: Connection,
    factory: EngineProcessFactory,
) -> None:
    """运行 Engine IPC server；异常会通知对端并继续向调用方传播。"""

    try:
        asyncio.run(serve_engine_connection(connection, factory))
    except BaseException as exc:
        with suppress(BrokenPipeError, EOFError, OSError):
            connection.send(_ProcessFailure(f"engine process failed: {type(exc).__name__}: {exc}"))
        raise
    finally:
        connection.close()


def _engine_process_main(
    connection: Connection,
    factory: EngineProcessFactory,
) -> None:
    try:
        run_engine_server(connection, factory)
    except BaseException:
        # 对端已经收到结构化失败；子进程只需结束，不重复打印 traceback。
        return


class ConnectionEngineClient:
    """通过一条已建立的双向连接访问远端 Engine。

    客户端不拥有远端进程，只负责请求流、取消、指标和关闭握手。进程由谁
    创建属于 composition root；普通 Engine 子进程和 torchrun rank 0 共用
    这一协议实现。
    """

    def __init__(
        self,
        connection: Connection,
        *,
        startup_timeout_seconds: float = 600.0,
        shutdown_timeout_seconds: float = 30.0,
        metrics_timeout_seconds: float = 5.0,
    ) -> None:
        self._connection = connection
        self._startup_timeout_seconds = startup_timeout_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._metrics_timeout_seconds = metrics_timeout_seconds
        self._capabilities = EngineCapabilities()
        self._snapshot: PerformanceSnapshot | None = None
        self._reader: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started_future: asyncio.Future[None] | None = None
        self._stopped = threading.Event()
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._request_ids = count(1)
        self._metric_call_ids = count(1)
        self._streams: dict[int, asyncio.Queue[GenerationEvent | GenerationError]] = {}
        self._metric_waiters: dict[int, _MetricsWaiter] = {}
        self._closing = False

    @property
    def ready(self) -> bool:
        return (
            not self._closing
            and self._started_future is not None
            and self._started_future.done()
            and self._started_future.exception() is None
            and not self._stopped.is_set()
        )

    @property
    def capabilities(self) -> EngineCapabilities:
        return self._capabilities

    async def start(self) -> None:
        if self._started_future is not None:
            await asyncio.shield(self._started_future)
            return
        self._loop = asyncio.get_running_loop()
        self._started_future = self._loop.create_future()
        self._reader = threading.Thread(
            target=self._read_responses,
            name="light-vllm-engine-response-reader",
            daemon=True,
        )
        self._reader.start()
        try:
            await asyncio.wait_for(
                asyncio.shield(self._started_future),
                timeout=self._startup_timeout_seconds,
            )
        except BaseException:
            await self.close()
            raise

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        generated_token_ids: list[int] = []
        finish_reason = None
        events = self.stream(request)
        try:
            async for event in events:
                if isinstance(event, TokenGenerated):
                    generated_token_ids.append(event.token_id)
                elif isinstance(event, GenerationFinished):
                    finish_reason = event.finish_reason
        finally:
            await events.aclose()
        if finish_reason is None:
            raise GenerationError("generation stream ended without a terminal event")
        return GenerateResult(
            input_ids=request.input_ids,
            generated_token_ids=tuple(generated_token_ids),
            finish_reason=finish_reason,
        )

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[GenerationEvent, None]:
        if not self.ready:
            raise GenerationNotReadyError("engine process is not ready")
        request_id = next(self._request_ids)
        queue: asyncio.Queue[GenerationEvent | GenerationError] = asyncio.Queue()
        self._streams[request_id] = queue
        finished = False
        self._send(_StartStream(request_id, request))
        try:
            while True:
                item = await queue.get()
                if isinstance(item, GenerationError):
                    raise item
                yield item
                if isinstance(item, GenerationFinished):
                    finished = True
                    return
        finally:
            self._streams.pop(request_id, None)
            if not finished and self.ready:
                self._send(_CancelStream(request_id))

    def snapshot(self) -> PerformanceSnapshot:
        if not self.ready:
            if self._snapshot is None:
                raise GenerationNotReadyError("engine process is not ready")
            return self._snapshot
        call_id = next(self._metric_call_ids)
        waiter = _MetricsWaiter(threading.Event())
        with self._state_lock:
            self._metric_waiters[call_id] = waiter
        try:
            self._send(_ReadMetrics(call_id))
            if not waiter.ready.wait(self._metrics_timeout_seconds):
                raise TimeoutError("timed out reading engine-process metrics")
            if waiter.error is not None:
                raise waiter.error
            assert waiter.snapshot is not None
            self._snapshot = waiter.snapshot
            return waiter.snapshot
        finally:
            with self._state_lock:
                self._metric_waiters.pop(call_id, None)

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if not self._stopped.is_set():
            with suppress(BrokenPipeError, EOFError, OSError):
                self._send(_Shutdown())
            await asyncio.to_thread(
                self._stopped.wait,
                self._shutdown_timeout_seconds,
            )
        self._connection.close()

    def _send(self, command: _Command) -> None:
        with self._send_lock:
            self._connection.send(command)

    def _read_responses(self) -> None:
        loop = self._loop
        assert loop is not None
        try:
            while True:
                response = self._connection.recv()
                loop.call_soon_threadsafe(self._dispatch, response)
                if isinstance(response, _Stopped) or (
                    isinstance(response, _ResponseBatch)
                    and any(isinstance(item, _Stopped) for item in response.responses)
                ):
                    return
        except (EOFError, OSError):
            # 客户端事件循环可能与连接同时关闭，reader 不应制造线程异常。
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(self._lost)

    def _dispatch(self, response: _Response) -> None:
        if isinstance(response, _ResponseBatch):
            for item in response.responses:
                self._dispatch(item)
            return
        if isinstance(response, _Started):
            self._capabilities = response.capabilities
            self._snapshot = response.snapshot
            assert self._started_future is not None
            if not self._started_future.done():
                self._started_future.set_result(None)
            return
        if isinstance(response, _StreamEvent):
            queue = self._streams.get(response.request_id)
            if queue is not None:
                queue.put_nowait(response.event)
            return
        if isinstance(response, _StreamFailure):
            queue = self._streams.get(response.request_id)
            if queue is not None:
                queue.put_nowait(_remote_error(response.kind, response.detail))
            return
        if isinstance(response, _MetricsResult):
            with self._state_lock:
                waiter = self._metric_waiters.get(response.call_id)
            if waiter is not None:
                waiter.snapshot = response.snapshot
                waiter.ready.set()
            return
        if isinstance(response, _ProcessFailure):
            self._fail(GenerationError(response.detail))
            return
        if isinstance(response, _Stopped):
            self._stopped.set()
            return
        self._fail(GenerationError(f"invalid engine-process response: {response!r}"))

    def _lost(self) -> None:
        if not self._closing:
            self._fail(GenerationError("engine process exited unexpectedly"))
        self._stopped.set()

    def _fail(self, error: GenerationError) -> None:
        started = self._started_future
        if started is not None and not started.done():
            started.set_exception(error)
        for queue in tuple(self._streams.values()):
            queue.put_nowait(error)
        with self._state_lock:
            waiters = tuple(self._metric_waiters.values())
        for waiter in waiters:
            waiter.error = error
            waiter.ready.set()


class ProcessEngineClient:
    """创建并访问独立 Engine 子进程。"""

    def __init__(
        self,
        factory: EngineProcessFactory,
        *,
        start_method: str = "spawn",
        startup_timeout_seconds: float = 600.0,
        shutdown_timeout_seconds: float = 30.0,
        metrics_timeout_seconds: float = 5.0,
    ) -> None:
        self._factory = factory
        self._context = multiprocessing.get_context(start_method)
        self._startup_timeout_seconds = startup_timeout_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._metrics_timeout_seconds = metrics_timeout_seconds
        self._client: ConnectionEngineClient | None = None
        self._process: multiprocessing.Process | None = None

    @property
    def ready(self) -> bool:
        process = self._process
        return (
            self._client is not None
            and self._client.ready
            and process is not None
            and process.is_alive()
        )

    @property
    def capabilities(self) -> EngineCapabilities:
        if self._client is None:
            return EngineCapabilities()
        return self._client.capabilities

    async def start(self) -> None:
        if self._client is not None:
            await self._client.start()
            return
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_engine_process_main,
            args=(child, self._factory),
            name="light-vllm-engine-process",
            daemon=True,
        )
        process.start()
        child.close()
        self._process = process
        self._client = ConnectionEngineClient(
            parent,
            startup_timeout_seconds=self._startup_timeout_seconds,
            shutdown_timeout_seconds=self._shutdown_timeout_seconds,
            metrics_timeout_seconds=self._metrics_timeout_seconds,
        )
        try:
            await self._client.start()
        except BaseException:
            await self.close()
            raise

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        return await self._require_client().generate(request)

    def stream(self, request: GenerateRequest) -> AsyncGenerator[GenerationEvent, None]:
        return self._require_client().stream(request)

    def snapshot(self) -> PerformanceSnapshot:
        return self._require_client().snapshot()

    async def close(self) -> None:
        process = self._process
        client = self._client
        if process is None:
            return
        if client is not None:
            await client.close()
        await asyncio.to_thread(process.join, self._shutdown_timeout_seconds)
        if process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join, self._shutdown_timeout_seconds)
        self._process = None

    def _require_client(self) -> ConnectionEngineClient:
        if self._client is None:
            raise GenerationNotReadyError("engine process is not started")
        return self._client
