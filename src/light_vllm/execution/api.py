from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ExecutionError(RuntimeError):
    """执行一次模型计算失败时抛出。"""


class ExecutionNotReadyError(ExecutionError):
    """模型执行器还没准备好时抛出。"""


@dataclass(frozen=True, slots=True)
class SequenceTokens:
    """一个请求当前已有的完整 token 序列。

    当前 P1 基线没有 KV cache，因此每轮执行都携带 prompt 和已生成 token。
    ``request_id`` 只用于把批量结果路由回 Engine 中的对应请求。
    """

    request_id: str
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        token_ids = tuple(self.token_ids)
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
            raise ValueError("token_ids must contain non-negative integers")
        object.__setattr__(self, "token_ids", token_ids)


@dataclass(frozen=True, slots=True)
class ExecutionBatch:
    """一次模型迭代需要执行的不可变序列集合。

    批次只描述“执行什么”，不携带 Scheduler 队列或 HTTP 状态。请求 ID
    必须唯一，执行器需要为每个 ID 返回且只返回一个 ``TokenSelection``。
    """

    sequences: tuple[SequenceTokens, ...]

    def __post_init__(self) -> None:
        sequences = tuple(self.sequences)
        if not sequences:
            raise ValueError("execution batch must not be empty")
        request_ids = tuple(sequence.request_id for sequence in sequences)
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("execution batch request IDs must be unique")
        object.__setattr__(self, "sequences", sequences)


@dataclass(frozen=True, slots=True)
class TokenSelection:
    """一次批量执行为某个请求选出的 token。

    显式携带 request ID，避免 Engine 依赖执行器的返回顺序进行结果路由。
    """

    request_id: str
    token_id: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if type(self.token_id) is not int or self.token_id < 0:
            raise ValueError("token_id must be a non-negative integer")


class TokenExecutor(Protocol):
    """单请求 reference 路径使用的逐 token 执行接口。"""

    @property
    def ready(self) -> bool: ...

    def next_token(self, token_ids: tuple[int, ...]) -> int:
        """根据一个请求的完整 token 序列计算下一个 token。"""

        ...


class BatchTokenExecutor(Protocol):
    """raw/continuous 路径共用的批量 token 执行接口。"""

    @property
    def ready(self) -> bool: ...

    def next_tokens(self, batch: ExecutionBatch) -> tuple[TokenSelection, ...]:
        """为批次中的每个 request ID 返回一个 token。"""

        ...
