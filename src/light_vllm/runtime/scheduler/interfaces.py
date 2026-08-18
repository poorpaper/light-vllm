from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from light_vllm.runtime.kv_cache import KVCacheStats


class SchedulerError(RuntimeError):
    """请求或调度状态违反契约时抛出。"""


@dataclass(frozen=True, slots=True)
class DecodingBudget:
    """算完现有 token 后，本轮还可以为生成新 token 预留多少资源。

    普通解码使用默认值：不提前留位置，最多返回一个新 token。投机解码可以
    提前留出多个位置，并允许执行器一次返回多个通过验证的 token。
    """

    num_lookahead_tokens: int = 0
    max_output_tokens: int = 1

    def __post_init__(self) -> None:
        if type(self.num_lookahead_tokens) is not int or self.num_lookahead_tokens < 0:
            raise ValueError("num_lookahead_tokens must be a non-negative integer")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")


@dataclass(frozen=True, slots=True)
class ShortRequestPolicy:
    """为首 token 工作量小且总长度受限的请求保留资源。

    预留 token 和 sequence 只用于短请求的首次输出；已经产生
    可见 token 的请求回到通用池，不长期占用 TTFT 保护资源。
    """

    max_effective_prompt_tokens: int
    max_total_tokens: int
    reserved_scheduled_tokens: int
    reserved_kv_token_slots: int
    reserved_sequences: int = 1
    regular_aging_steps: int = 8

    def __post_init__(self) -> None:
        values = (
            self.max_effective_prompt_tokens,
            self.max_total_tokens,
            self.reserved_scheduled_tokens,
            self.reserved_sequences,
            self.regular_aging_steps,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("short-request limits and reserves must be positive integers")
        if type(self.reserved_kv_token_slots) is not int or self.reserved_kv_token_slots < 0:
            raise ValueError("reserved_kv_token_slots must be a non-negative integer")
        if self.reserved_scheduled_tokens < self.max_effective_prompt_tokens:
            raise ValueError("short token reserve must cover one eligible effective prompt")
        if self.reserved_kv_token_slots < self.max_total_tokens - 1:
            raise ValueError("short KV reserve must cover one eligible request")


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    """Scheduler 为一个请求安排的本轮工作量。

    ``num_scheduled_tokens`` 是本轮要计算的已知 token 数；
    ``num_lookahead_tokens`` 是为投机解码提前留出的未知 token 位置数。
    block table 必须覆盖两部分可能写入的全部 KV cache。
    """

    request_id: str
    num_computed_tokens: int
    num_scheduled_tokens: int
    # 这里只预留位置；候选 token 由执行器产生。
    num_lookahead_tokens: int
    # 执行器本轮最多可以返回多少个通过验证的新 token。
    max_output_tokens: int
    # 分页后端返回 block table；连续缓存返回 None。
    block_ids: tuple[int, ...] | None
    # 完整写入且不会再修改的前缀页数；只有这些页可以跨请求共享。
    num_readonly_prefix_blocks: int = 0

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
        if type(self.num_readonly_prefix_blocks) is not int or self.num_readonly_prefix_blocks < 0:
            raise ValueError("num_readonly_prefix_blocks must be a non-negative integer")
        if self.block_ids is None and self.num_readonly_prefix_blocks:
            raise ValueError("readonly prefix blocks require a block table")
        if self.block_ids is not None:
            object.__setattr__(self, "block_ids", tuple(self.block_ids))
            if self.num_readonly_prefix_blocks > len(self.block_ids):
                raise ValueError("readonly prefix blocks must fit within the block table")


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    """Scheduler 一轮调度选出的请求和工作量。"""

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


@dataclass(frozen=True, slots=True)
class SchedulerStats:
    """Scheduler 对观测面公开的不可变负载快照。

    pending tokens 只统计当前已知但尚未计算的输入；max remaining tokens
    还包含请求允许生成的最大输出，是适合 HPA 的保守工作量上界。
    """

    waiting_requests: int
    running_requests: int
    waiting_pending_tokens: int
    running_pending_tokens: int
    waiting_max_remaining_tokens: int
    running_max_remaining_tokens: int
    kv_cache: KVCacheStats

    def __post_init__(self) -> None:
        values = (
            self.waiting_requests,
            self.running_requests,
            self.waiting_pending_tokens,
            self.running_pending_tokens,
            self.waiting_max_remaining_tokens,
            self.running_max_remaining_tokens,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("scheduler statistics must be non-negative integers")


class Scheduler(Protocol):
    """根据单轮 token 上限和 KV cache 容量安排每次模型计算。"""

    @property
    def has_requests(self) -> bool: ...

    @property
    def max_num_sequences(self) -> int: ...

    @property
    def max_num_scheduled_tokens(self) -> int: ...

    @property
    def stats(self) -> SchedulerStats: ...

    def add(
        self,
        request_id: str,
        *,
        token_ids: tuple[int, ...],
        max_num_tokens: int,
        cache_epoch: int | None = None,
    ) -> None: ...

    def remove(self, request_id: str) -> bool: ...

    def schedule(self) -> SchedulerOutput: ...

    def complete(
        self,
        request_id: str,
        *,
        num_committed_tokens: int,
        num_new_tokens: int,
    ) -> None: ...
