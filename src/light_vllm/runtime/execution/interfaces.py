from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ExecutionError(RuntimeError):
    """执行一次模型计算失败时抛出。"""


class ExecutionNotReadyError(ExecutionError):
    """模型执行器还没有准备好时抛出。"""


@dataclass(frozen=True, slots=True)
class ExecutionCapabilities:
    """一个执行拓扑初始化后可提供的模型与 KV 容量。"""

    max_model_tokens: int | None
    max_kv_cache_tokens: int | None


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """一个请求在本轮真正交给模型计算的事实型输入。

    ``input_token_ids`` 始终是已知输入；lookahead 只有容量和位置，没有伪造
    token ID。投机 Executor 应自行产生 proposal，并在输出中报告确认结果。
    """

    request_id: str
    input_token_ids: tuple[int, ...]
    num_computed_tokens: int
    num_lookahead_tokens: int
    max_output_tokens: int
    # 只把分页后端需要的 block table 传给 Executor。
    block_ids: tuple[int, ...] | None

    def __post_init__(self) -> None:
        input_token_ids = tuple(self.input_token_ids)
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not input_token_ids:
            raise ValueError("input_token_ids must not be empty")
        if any(type(token_id) is not int or token_id < 0 for token_id in input_token_ids):
            raise ValueError("input_token_ids must contain non-negative integers")
        if type(self.num_computed_tokens) is not int or self.num_computed_tokens < 0:
            raise ValueError("num_computed_tokens must be a non-negative integer")
        if type(self.num_lookahead_tokens) is not int or self.num_lookahead_tokens < 0:
            raise ValueError("num_lookahead_tokens must be a non-negative integer")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 0:
            raise ValueError("max_output_tokens must be a non-negative integer")
        object.__setattr__(self, "input_token_ids", input_token_ids)
        if self.block_ids is not None:
            object.__setattr__(self, "block_ids", tuple(self.block_ids))


@dataclass(frozen=True, slots=True)
class ExecutionBatch:
    """SchedulerOutput 转换得到的一次不可变模型执行输入。"""

    requests: tuple[ExecutionRequest, ...]

    def __post_init__(self) -> None:
        requests = tuple(self.requests)
        request_ids = tuple(request.request_id for request in requests)
        if not requests:
            raise ValueError("execution batch must not be empty")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("execution batch request IDs must be unique")
        object.__setattr__(self, "requests", requests)

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self.requests)


@dataclass(frozen=True, slots=True)
class RequestOutput:
    """一个请求本轮完成的输入计算和最终确认 token。

    ``output_token_ids`` 的前 ``num_cached_output_tokens`` 个 token 已经写入
    KV；其余 token 下一轮仍需作为输入计算。普通 decode 通常返回一个未
    缓存 token；投机验证可以返回多个 token，并缓存其中已验证的 draft 前缀。
    """

    request_id: str
    num_input_tokens_computed: int
    output_token_ids: tuple[int, ...] = ()
    num_cached_output_tokens: int = 0

    def __post_init__(self) -> None:
        output_token_ids = tuple(self.output_token_ids)
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if type(self.num_input_tokens_computed) is not int or self.num_input_tokens_computed < 0:
            raise ValueError("num_input_tokens_computed must be a non-negative integer")
        if any(type(token_id) is not int or token_id < 0 for token_id in output_token_ids):
            raise ValueError("output_token_ids must contain non-negative integers")
        if type(
            self.num_cached_output_tokens
        ) is not int or not 0 <= self.num_cached_output_tokens <= len(output_token_ids):
            raise ValueError("num_cached_output_tokens must be within output_token_ids")
        object.__setattr__(self, "output_token_ids", output_token_ids)


@dataclass(frozen=True, slots=True)
class ExecutionOutput:
    """一次模型执行中每个请求的独立结果。"""

    requests: tuple[RequestOutput, ...]

    def __post_init__(self) -> None:
        requests = tuple(self.requests)
        request_ids = tuple(request.request_id for request in requests)
        if not requests:
            raise ValueError("execution output must not be empty")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("execution output request IDs must be unique")
        object.__setattr__(self, "requests", requests)


class ExecutionLease(Protocol):
    """保证执行期间请求物理资源仍然有效的幂等租约。"""

    def release(self) -> None: ...


class TokenExecutionSession(Protocol):
    """reference 请求固定模型后使用的逐 token 接口。"""

    def next_token(self, token_ids: tuple[int, ...]) -> int: ...


class TokenExecutor(Protocol):
    """为单个 reference 请求创建固定模型的执行会话。"""

    @property
    def ready(self) -> bool: ...

    def open_session(self) -> TokenExecutionSession: ...


class ModelExecutor(Protocol):
    """Engine Core 驱动的执行拓扑边界，可协调一个或多个 Worker。"""

    @property
    def ready(self) -> bool: ...

    @property
    def capabilities(self) -> ExecutionCapabilities: ...

    def initialize(self) -> None: ...

    def add_request(self, request_id: str, *, capacity: int) -> None: ...

    def free_request(self, request_id: str) -> bool: ...

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease: ...

    def execute(self, batch: ExecutionBatch) -> ExecutionOutput: ...


class ModelWorker(Protocol):
    """一个设备/rank 内拥有模型计算和物理缓存的同步执行单元。"""

    @property
    def ready(self) -> bool: ...

    @property
    def capabilities(self) -> ExecutionCapabilities: ...

    def initialize(self) -> None: ...

    def add_request(self, request_id: str, *, capacity: int) -> None: ...

    def free_request(self, request_id: str) -> bool: ...

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease: ...

    def execute(self, batch: ExecutionBatch) -> ExecutionOutput: ...
