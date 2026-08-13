from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SchedulerError(RuntimeError):
    """请求状态违反 Scheduler 契约时抛出。"""


@dataclass(frozen=True, slots=True)
class SchedulerBatch:
    """某次模型迭代应执行的请求 ID。

    Scheduler 只返回 ID，不复制 token 或生成参数。Engine 用这些 ID 从自己
    独占的请求状态中构造 ``ExecutionBatch``，从而避免 Scheduler 接触执行细节。
    """

    request_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        request_ids = tuple(self.request_ids)
        if any(not request_id for request_id in request_ids):
            raise ValueError("scheduler request IDs must not be empty")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("scheduler request IDs must be unique")
        object.__setattr__(self, "request_ids", request_ids)


class Scheduler(Protocol):
    """在每次模型迭代前选择本轮运行请求。

    Scheduler 拥有“等待/运行”的成员关系，Engine 拥有请求内容。所有方法由
    Engine 在自己的状态锁内调用，因此实现本身不需要再管理并发锁。
    """

    @property
    def has_requests(self) -> bool:
        """是否还有等待或运行中的请求。"""

        ...

    def add(self, request_id: str) -> None:
        """把新请求加入等待集合；重复 ID 必须报错。"""

        ...

    def remove(self, request_id: str) -> bool:
        """从等待或运行集合移除请求，并返回此前是否存在。"""

        ...

    def schedule(self) -> SchedulerBatch:
        """在迭代安全点选择当前执行批次。"""

        ...
