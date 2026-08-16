from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

    from light_vllm.modeling.models.interfaces import ModelSession


class ExecutionError(RuntimeError):
    """执行一次模型计算失败时抛出。"""


class ExecutionNotReadyError(ExecutionError):
    """模型执行器还没有准备好时抛出。"""


@dataclass(frozen=True, slots=True)
class ExecutionCapabilities:
    """执行器初始化后能够支持的模型长度和 KV cache 容量。"""

    max_model_tokens: int | None
    max_kv_cache_tokens: int | None
    # 物理 KV 重新创建时递增；prefix cache 不能跨 epoch 复用旧页。
    kv_cache_epoch: int | None = None


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """一个请求本轮交给模型计算的输入。

    ``input_token_ids`` 是已经确定的 token。``num_lookahead_tokens`` 只表示
    为投机解码提前留出的 KV 位置；这些位置的 token 还没有确定，由执行器
    产生候选 token 并返回最终通过验证的结果。
    """

    request_id: str
    input_token_ids: tuple[int, ...]
    num_computed_tokens: int
    num_lookahead_tokens: int
    max_output_tokens: int
    # 分页 KV cache 需要 block table；连续 KV cache 不需要，使用 None。
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
    """一次模型调用需要执行的所有请求。"""

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
    """一个请求本轮算完了多少输入，以及最终确认了哪些新 token。

    ``output_token_ids`` 的前 ``num_cached_output_tokens`` 个 token 已经写入
    KV cache；其余 token 虽然已经确认，但下一轮仍要作为输入再计算一次。
    普通解码通常返回一个尚未写入缓存的 token；投机解码可以一次返回多个。
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
    """保证模型计算结束前，请求使用的缓存不会被释放或复用。"""

    def release(self) -> None: ...


class TokenExecutionSession(Protocol):
    """参考实现选定模型后，用来逐个计算新 token 的接口。"""

    def next_token(self, token_ids: tuple[int, ...]) -> int: ...


class TokenExecutor(Protocol):
    """为一次参考生成选定模型，并创建逐 token 执行对象。"""

    @property
    def ready(self) -> bool: ...

    def open_session(self) -> TokenExecutionSession: ...


class ModelExecutor(Protocol):
    """Engine 调用模型的统一入口。

    Executor 隐藏本进程、多进程或分布式等执行方式，并可以协调多个 Worker。
    """

    @property
    def ready(self) -> bool: ...

    @property
    def capabilities(self) -> ExecutionCapabilities: ...

    def initialize(self) -> None: ...

    def add_request(self, request_id: str, *, capacity: int) -> None: ...

    def free_request(self, request_id: str) -> bool: ...

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease: ...

    def execute(self, batch: ExecutionBatch) -> ExecutionOutput: ...


class ModelStepHandler(Protocol):
    """执行一次已经确定 token 的模型计算，并管理对应的物理 KV cache。"""

    @property
    def max_kv_cache_tokens(self) -> int | None: ...

    def add_request(self, request_id: str, *, capacity: int) -> None: ...

    def free_request(self, request_id: str) -> None: ...

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease: ...

    def forward(
        self,
        model: ModelSession,
        batch: ExecutionBatch,
    ) -> tuple[Tensor, ...]:
        """返回每个请求有效位置的 ``[query, vocabulary]`` logits。"""

        ...

    def truncate(self, request_id: str, num_cached_tokens: int) -> None:
        """丢弃尚未确认的物理 KV 尾部；没有长度状态的实现可以不处理。"""

        ...


class DecodeHandler(Protocol):
    """组织无输出输入步骤、普通解码或投机解码，并产出确认 token。"""

    def execute(
        self,
        model: ModelSession,
        batch: ExecutionBatch,
        step: ModelStepHandler,
    ) -> ExecutionOutput: ...


class ModelWorker(Protocol):
    """一个设备或分布式 rank 内的模型执行单元。"""

    @property
    def ready(self) -> bool: ...

    @property
    def capabilities(self) -> ExecutionCapabilities: ...

    def initialize(self) -> None: ...

    def add_request(self, request_id: str, *, capacity: int) -> None: ...

    def free_request(self, request_id: str) -> bool: ...

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease: ...

    def execute(self, batch: ExecutionBatch) -> ExecutionOutput: ...
