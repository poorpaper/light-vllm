from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SchedulerError(RuntimeError):
    """请求或调度状态违反契约时抛出。"""


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    """一个请求在本轮获得的 token 预算和逻辑 KV block table。"""

    request_id: str
    num_computed_tokens: int
    num_scheduled_tokens: int
    block_ids: tuple[int, ...]
    sampling_required: bool

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if type(self.num_computed_tokens) is not int or self.num_computed_tokens < 0:
            raise ValueError("num_computed_tokens must be a non-negative integer")
        if type(self.num_scheduled_tokens) is not int or self.num_scheduled_tokens <= 0:
            raise ValueError("num_scheduled_tokens must be a positive integer")
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

    def add(self, request_id: str, *, num_tokens: int) -> None: ...

    def remove(self, request_id: str) -> bool: ...

    def schedule(self) -> SchedulerOutput: ...

    def complete(
        self,
        request_id: str,
        *,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> None: ...
