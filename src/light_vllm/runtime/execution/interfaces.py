from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ExecutionError(RuntimeError):
    """执行一次模型计算失败时抛出。"""


class ExecutionNotReadyError(ExecutionError):
    """模型执行器还没有准备好时抛出。"""


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """一个请求在本轮真正交给模型计算的 token 切片和可选 KV block table。"""

    request_id: str
    input_token_ids: tuple[int, ...]
    num_computed_tokens: int
    # 只把分页后端需要的 block table 传给 Executor。
    block_ids: tuple[int, ...] | None
    sampling_required: bool

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
    """一个请求本轮完成的缓存计算和最终确认 token。

    ``token_ids`` 可以为空，也可以包含多个 token：chunked prefill 不产生
    token，普通 decode 产生一个，未来投机解码可以一次确认多个。
    """

    request_id: str
    num_computed_tokens: int
    token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        token_ids = tuple(self.token_ids)
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if type(self.num_computed_tokens) is not int or self.num_computed_tokens < 0:
            raise ValueError("num_computed_tokens must be a non-negative integer")
        if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
            raise ValueError("token_ids must contain non-negative integers")
        object.__setattr__(self, "token_ids", token_ids)


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


class TokenExecutor(Protocol):
    """单请求 reference 路径使用的逐 token 接口。"""

    @property
    def ready(self) -> bool: ...

    def next_token(self, token_ids: tuple[int, ...]) -> int: ...


class ModelExecutor(Protocol):
    """Engine Core 驱动的模型执行端口。"""

    @property
    def ready(self) -> bool: ...

    def add_request(self, request_id: str, *, capacity: int) -> None: ...

    def free_request(self, request_id: str) -> bool: ...

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease: ...

    def execute(self, batch: ExecutionBatch) -> ExecutionOutput: ...


class ModelWorker(Protocol):
    """一个设备 rank 内拥有模型计算和物理缓存的同步执行单元。"""

    @property
    def ready(self) -> bool: ...

    def add_request(self, request_id: str, *, capacity: int) -> None: ...

    def free_request(self, request_id: str) -> bool: ...

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease: ...

    def execute(self, batch: ExecutionBatch) -> ExecutionOutput: ...
