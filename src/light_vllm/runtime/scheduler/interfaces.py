from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SchedulerError(RuntimeError):
    """请求或调度状态违反契约时抛出。"""


@dataclass(frozen=True, slots=True)
class DecodingBudget:
    """输出 frontier 上一次执行可额外使用的资源预算。

    它描述资源事实，不把请求标记成 prefill、decode 或 speculative 模式。
    """

    num_lookahead_tokens: int = 0
    max_output_tokens: int = 1

    def __post_init__(self) -> None:
        if type(self.num_lookahead_tokens) is not int or self.num_lookahead_tokens < 0:
            raise ValueError("num_lookahead_tokens must be a non-negative integer")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    """一个请求在本轮获得的事实型执行预算。

    ``num_scheduled_tokens`` 是已有明确 token ID 的输入；lookahead 只预留
    尚无 token ID 的投机位置。block table 覆盖两者可能写入的完整 KV 范围。
    """

    request_id: str
    num_computed_tokens: int
    num_scheduled_tokens: int
    # 仅表示已预留的未知输出位置；proposal 的产生属于执行侧。
    num_lookahead_tokens: int
    # Executor 本轮最多可以返回多少个最终确认的输出 token。
    max_output_tokens: int
    # 分页后端返回 block table；连续缓存返回 None。
    block_ids: tuple[int, ...] | None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if type(self.num_computed_tokens) is not int or self.num_computed_tokens < 0:
            raise ValueError("num_computed_tokens must be a non-negative integer")
        if type(self.num_scheduled_tokens) is not int or self.num_scheduled_tokens <= 0:
            raise ValueError("num_scheduled_tokens must be a positive integer")
        if type(self.num_lookahead_tokens) is not int or self.num_lookahead_tokens < 0:
            raise ValueError("num_lookahead_tokens must be a non-negative integer")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 0:
            raise ValueError("max_output_tokens must be a non-negative integer")
        if self.block_ids is not None:
            object.__setattr__(self, "block_ids", tuple(self.block_ids))


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    """Scheduler 在一个安全点产生的不可变执行计划。"""

    requests: tuple[ScheduledRequest, ...]

    def __post_init__(self) -> None:
        requests = tuple(self.requests)
        request_ids = tuple(request.request_id for request in requests)
        if not requests:
            raise ValueError("scheduler output must not be empty")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("scheduler output request IDs must be unique")
        object.__setattr__(self, "requests", requests)

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self.requests)


class Scheduler(Protocol):
    """根据 token budget 和逻辑 KV 容量规划每次模型执行。"""

    @property
    def has_requests(self) -> bool: ...

    @property
    def max_num_sequences(self) -> int: ...

    @property
    def max_num_scheduled_tokens(self) -> int: ...

    def add(self, request_id: str, *, num_tokens: int) -> None: ...

    def remove(self, request_id: str) -> bool: ...

    def schedule(self) -> SchedulerOutput: ...

    def complete(
        self,
        request_id: str,
        *,
        num_committed_tokens: int,
        num_new_tokens: int,
    ) -> None: ...
