"""FCFS token-budget Scheduler。"""

from __future__ import annotations

from collections import deque
from contextlib import suppress
from dataclasses import dataclass

from light_vllm.runtime.kv_cache import KVCacheCapacityError, KVCacheManager
from light_vllm.runtime.scheduler.interfaces import (
    DecodingBudget,
    ScheduledRequest,
    SchedulerError,
    SchedulerOutput,
)


@dataclass(slots=True)
class _RequestState:
    num_tokens: int
    prompt_token_ids: tuple[int, ...]
    cache_epoch: int | None
    num_computed_tokens: int = 0
    kv_attached: bool = False


class TokenBudgetScheduler:
    """在固定 token budget 内按 FCFS 选择请求。

    prompt 和输出 token 使用同一套计数。长 prompt 会自然拆成多个 chunk；
    只有追上当前全部已知 token 时，执行侧才需要采样新 token。
    """

    def __init__(
        self,
        kv_cache: KVCacheManager,
        *,
        max_num_sequences: int,
        max_num_scheduled_tokens: int,
        decoding_budget: DecodingBudget | None = None,
    ) -> None:
        if type(max_num_sequences) is not int or max_num_sequences <= 0:
            raise ValueError("max_num_sequences must be a positive integer")
        if type(max_num_scheduled_tokens) is not int or max_num_scheduled_tokens <= 0:
            raise ValueError("max_num_scheduled_tokens must be a positive integer")
        self._kv_cache = kv_cache
        self._max_num_sequences = max_num_sequences
        self._max_num_scheduled_tokens = max_num_scheduled_tokens
        self._decoding_budget = decoding_budget or DecodingBudget()
        self._waiting: deque[str] = deque()
        self._running: dict[str, _RequestState] = {}
        self._states: dict[str, _RequestState] = {}

    @property
    def has_requests(self) -> bool:
        return bool(self._states)

    @property
    def max_num_sequences(self) -> int:
        return self._max_num_sequences

    @property
    def max_num_scheduled_tokens(self) -> int:
        return self._max_num_scheduled_tokens

    def add(
        self,
        request_id: str,
        *,
        token_ids: tuple[int, ...],
        cache_epoch: int | None = None,
    ) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        token_ids = tuple(token_ids)
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
            raise ValueError("token_ids must contain non-negative integers")
        if request_id in self._states:
            raise SchedulerError(f"request {request_id!r} is already scheduled")

        self._states[request_id] = _RequestState(
            num_tokens=len(token_ids),
            prompt_token_ids=token_ids,
            cache_epoch=cache_epoch,
        )
        self._waiting.append(request_id)

    def remove(self, request_id: str) -> bool:
        state = self._states.pop(request_id, None)
        if state is None:
            return False
        self._running.pop(request_id, None)
        with suppress(ValueError):
            self._waiting.remove(request_id)
        if state.kv_attached:
            self._kv_cache.free(request_id)
        return True

    def schedule(self) -> SchedulerOutput:
        self._fill_open_slots()
        token_budget = self._max_num_scheduled_tokens
        scheduled: list[ScheduledRequest] = []

        for request_id, state in self._running.items():
            if token_budget == 0:
                break
            # 这些 token 已经确定，但还没有算完并写入 KV cache。
            pending_tokens = state.num_tokens - state.num_computed_tokens
            if pending_tokens <= 0:
                raise SchedulerError(f"request {request_id!r} has no pending tokens")

            num_scheduled_tokens = min(pending_tokens, token_budget)
            num_lookahead_tokens = 0
            max_output_tokens = 0
            # 只有本轮把现有 token 全部算完，才可以继续生成新 token。
            if num_scheduled_tokens == pending_tokens:
                desired_lookahead = self._decoding_budget.num_lookahead_tokens
                if num_scheduled_tokens + desired_lookahead <= token_budget:
                    num_lookahead_tokens = desired_lookahead
                    max_output_tokens = self._decoding_budget.max_output_tokens
                elif pending_tokens == 1:
                    # 资源紧张时退化为普通 decode，保证请求仍能前进。
                    max_output_tokens = 1
                else:
                    # 当前预算不够时留一个 token 到下一轮，届时再申请投机位置。
                    num_scheduled_tokens -= 1

            # 投机位置也可能写入 KV cache，因此必须和现有输入一起预留空间。
            num_reserved_tokens = num_scheduled_tokens + num_lookahead_tokens
            try:
                reservation = self._kv_cache.reserve(request_id, num_reserved_tokens)
            except KVCacheCapacityError:
                # 其他运行请求可能在本轮完成并释放 block；暂时跳过即可。
                continue

            scheduled.append(
                ScheduledRequest(
                    request_id=request_id,
                    num_computed_tokens=state.num_computed_tokens,
                    num_scheduled_tokens=num_scheduled_tokens,
                    num_lookahead_tokens=num_lookahead_tokens,
                    max_output_tokens=max_output_tokens,
                    block_ids=reservation.block_ids,
                )
            )
            token_budget -= num_reserved_tokens

        if not scheduled:
            raise SchedulerError("no request fits the token budget and available KV cache")
        return SchedulerOutput(requests=tuple(scheduled))

    def complete(
        self,
        request_id: str,
        *,
        num_committed_tokens: int,
        num_new_tokens: int,
    ) -> None:
        try:
            state = self._states[request_id]
        except KeyError as exc:
            raise SchedulerError(f"request {request_id!r} is not scheduled") from exc
        if type(num_committed_tokens) is not int or num_committed_tokens < 0:
            raise ValueError("num_committed_tokens must be a non-negative integer")
        if type(num_new_tokens) is not int or num_new_tokens < 0:
            raise ValueError("num_new_tokens must be a non-negative integer")

        # 提交已算完的输入，以及投机解码中已经写入缓存的新 token；其余预留释放。
        self._kv_cache.commit(request_id, num_committed_tokens)
        state.num_computed_tokens += num_committed_tokens
        # 尚未写入缓存的新 token 会在下一轮作为普通输入继续计算。
        state.num_tokens += num_new_tokens

    def _fill_open_slots(self) -> None:
        while self._waiting and len(self._running) < self._max_num_sequences:
            request_id = self._waiting.popleft()
            state = self._states[request_id]
            match = self._kv_cache.add_request(
                request_id,
                token_ids=state.prompt_token_ids,
                cache_epoch=state.cache_epoch,
            )
            if not 0 <= match.num_cached_tokens < len(state.prompt_token_ids):
                self._kv_cache.free(request_id)
                raise SchedulerError("cached prefix must leave at least one token to compute")
            state.num_computed_tokens = match.num_cached_tokens
            state.kv_attached = True
            self._running[request_id] = state
