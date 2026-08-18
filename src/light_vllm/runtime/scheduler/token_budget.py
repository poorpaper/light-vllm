"""带短请求保护和可选自回滚的分层 token-budget Scheduler。"""

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
    SchedulerStats,
    SelfResubmitPolicy,
    ShortRequestPolicy,
)


@dataclass(slots=True)
class _RequestState:
    num_tokens: int
    max_num_tokens: int
    prompt_token_ids: tuple[int, ...]
    cache_epoch: int | None
    num_computed_tokens: int = 0
    kv_attached: bool = False
    is_short: bool = False
    uses_short_admission_slot: bool = False
    has_visible_output: bool = False
    waiting_steps: int = 0
    has_completion_claim: bool = False
    force_completion_claim: bool = False
    num_resubmits: int = 0
    num_recomputed_tokens: int = 0


class TokenBudgetScheduler:
    """在固定 token budget 内按资源池和轮转顺序选择请求。

    prompt 和输出 token 使用同一套计数。长 prompt 会自然拆成多个 chunk；
    只有追上当前全部已知 token 时，执行侧才需要采样新 token。短请求策略
    关闭时，所有请求退化为一个通用池。
    """

    def __init__(
        self,
        kv_cache: KVCacheManager,
        *,
        max_num_sequences: int,
        max_num_scheduled_tokens: int,
        decoding_budget: DecodingBudget | None = None,
        short_request_policy: ShortRequestPolicy | None = None,
        self_resubmit_policy: SelfResubmitPolicy | None = None,
    ) -> None:
        if type(max_num_sequences) is not int or max_num_sequences <= 0:
            raise ValueError("max_num_sequences must be a positive integer")
        if type(max_num_scheduled_tokens) is not int or max_num_scheduled_tokens <= 0:
            raise ValueError("max_num_scheduled_tokens must be a positive integer")
        if short_request_policy is not None:
            if short_request_policy.reserved_sequences >= max_num_sequences:
                raise ValueError("short sequence reserve must leave a common sequence slot")
            if short_request_policy.reserved_scheduled_tokens >= max_num_scheduled_tokens:
                raise ValueError("short token reserve must leave a common token budget")
        self._kv_cache = kv_cache
        self._max_num_sequences = max_num_sequences
        self._max_num_scheduled_tokens = max_num_scheduled_tokens
        self._decoding_budget = decoding_budget or DecodingBudget()
        self._short_request_policy = short_request_policy
        self._self_resubmit_policy = self_resubmit_policy
        self._waiting: deque[str] = deque()
        self._running: dict[str, _RequestState] = {}
        self._short_launch_order: deque[str] = deque()
        self._common_order: deque[str] = deque()
        self._states: dict[str, _RequestState] = {}
        self._resubmitted_in_step: list[str] = []
        self._self_resubmits_total = 0
        self._self_resubmit_recomputed_tokens_total = 0

    @property
    def has_requests(self) -> bool:
        return bool(self._states)

    @property
    def max_num_sequences(self) -> int:
        return self._max_num_sequences

    @property
    def max_num_scheduled_tokens(self) -> int:
        return self._max_num_scheduled_tokens

    @property
    def stats(self) -> SchedulerStats:
        """返回一次性快照，不把内部可变队列暴露给 observer。"""

        return SchedulerStats(
            waiting_requests=len(self._waiting),
            running_requests=len(self._running),
            waiting_pending_tokens=sum(
                self._pending_tokens(self._states[request_id]) for request_id in self._waiting
            ),
            running_pending_tokens=sum(
                self._pending_tokens(state) for state in self._running.values()
            ),
            waiting_max_remaining_tokens=sum(
                self._max_remaining_tokens(self._states[request_id]) for request_id in self._waiting
            ),
            running_max_remaining_tokens=sum(
                self._max_remaining_tokens(state) for state in self._running.values()
            ),
            kv_cache=self._kv_cache.stats,
            short_launch_requests=len(self._short_launch_order),
            self_resubmits_total=self._self_resubmits_total,
            self_resubmit_recomputed_tokens_total=self._self_resubmit_recomputed_tokens_total,
        )

    def add(
        self,
        request_id: str,
        *,
        token_ids: tuple[int, ...],
        max_num_tokens: int,
        cache_epoch: int | None = None,
    ) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        token_ids = tuple(token_ids)
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
            raise ValueError("token_ids must contain non-negative integers")
        if type(max_num_tokens) is not int or max_num_tokens <= len(token_ids):
            raise ValueError("max_num_tokens must leave room for at least one output token")
        if request_id in self._states:
            raise SchedulerError(f"request {request_id!r} is already scheduled")

        self._states[request_id] = _RequestState(
            num_tokens=len(token_ids),
            max_num_tokens=max_num_tokens,
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
        for order in (self._short_launch_order, self._common_order):
            with suppress(ValueError):
                order.remove(request_id)
        if state.kv_attached:
            self._kv_cache.free(request_id)
        return True

    def schedule(self) -> SchedulerOutput:
        self._resubmitted_in_step.clear()
        self._rebalance_admission_slots()
        classifications = self._classify_waiting_requests()
        self._age_waiting_requests(classifications)
        self._admit_waiting_requests(classifications)
        scheduled = self._schedule_admitted_requests()

        if not scheduled and self._resubmitted_in_step:
            # 全部候选都撞墙时，让最早回滚者改走 strict claim，保证下一步能前进。
            recovery_id = self._resubmitted_in_step[0]
            recovery = self._states[recovery_id]
            recovery.force_completion_claim = True
            self._waiting.remove(recovery_id)
            self._waiting.appendleft(recovery_id)
            classifications = self._classify_waiting_requests()
            self._admit_waiting_requests(classifications)
            scheduled = self._schedule_admitted_requests()

        if not scheduled:
            raise SchedulerError("no admitted request can make progress")
        return SchedulerOutput(requests=tuple(scheduled))

    def _schedule_admitted_requests(self) -> list[ScheduledRequest]:
        scheduled: list[ScheduledRequest] = []

        policy = self._short_request_policy
        short_token_budget = 0 if policy is None else policy.reserved_scheduled_tokens
        short_sequence_budget = 0 if policy is None else policy.reserved_sequences

        # 短请求预留池只帮助尚未产生首 token 的请求。
        for request_id in self._rotate_request_ids(self._short_launch_order):
            if short_token_budget == 0 or short_sequence_budget == 0:
                break
            state = self._running[request_id]
            item, consumed = self._schedule_request(
                request_id,
                state,
                short_token_budget,
                prioritize_first_output=True,
            )
            if item is None:
                continue
            scheduled.append(item)
            short_token_budget -= consumed
            short_sequence_budget -= 1

        common_token_budget = self._max_num_scheduled_tokens - (
            0 if policy is None else policy.reserved_scheduled_tokens
        )
        common_sequence_budget = self._max_num_sequences - (
            0 if policy is None else policy.reserved_sequences
        )

        # 通用池使用 round-robin，避免队首长 prefill 连续吃完每轮预算。
        for request_id in self._rotate_request_ids(self._common_order):
            if common_token_budget == 0 or common_sequence_budget == 0:
                break
            state = self._running[request_id]
            item, consumed = self._schedule_request(request_id, state, common_token_budget)
            if item is None:
                continue
            scheduled.append(item)
            common_token_budget -= consumed
            common_sequence_budget -= 1

        return scheduled

    def _schedule_request(
        self,
        request_id: str,
        state: _RequestState,
        token_budget: int,
        *,
        prioritize_first_output: bool = False,
    ) -> tuple[ScheduledRequest | None, int]:
        if token_budget <= 0:
            return None, 0

        # 这些 token 已经确定，但还没有算完并写入 KV cache。
        pending_tokens = state.num_tokens - state.num_computed_tokens
        if pending_tokens <= 0:
            raise SchedulerError(f"request {request_id!r} has no pending tokens")

        num_scheduled_tokens = min(pending_tokens, token_budget)
        num_lookahead_tokens = 0
        max_output_tokens = 0
        # 只有本轮把现有 token 全部算完，才可以继续生成新 token。
        if num_scheduled_tokens == pending_tokens:
            remaining_output_tokens = state.max_num_tokens - state.num_tokens
            if remaining_output_tokens <= 0:
                raise SchedulerError(f"request {request_id!r} reached its token limit")
            round_max_output_tokens = min(
                self._decoding_budget.max_output_tokens,
                remaining_output_tokens,
            )
            # 最后一个确认 token 不会写入 KV，所以最多只需为其余输出预留位置。
            desired_lookahead = min(
                self._decoding_budget.num_lookahead_tokens,
                round_max_output_tokens - 1,
            )
            if num_scheduled_tokens + desired_lookahead <= token_budget:
                num_lookahead_tokens = desired_lookahead
                max_output_tokens = round_max_output_tokens
            elif pending_tokens == 1 or prioritize_first_output:
                # 短请求优先在本轮产出首 token，不为投机 lookahead 多等一步。
                max_output_tokens = 1
            else:
                # 当前预算不够时留一个 token 到下一轮，届时再申请投机位置。
                num_scheduled_tokens -= 1

        # 投机位置也可能写入 KV cache，因此必须和现有输入一起预留空间。
        num_reserved_tokens = num_scheduled_tokens + num_lookahead_tokens
        try:
            reservation = self._kv_cache.reserve(request_id, num_reserved_tokens)
        except KVCacheCapacityError as exc:
            if self._self_resubmit_policy is not None and not state.has_completion_claim:
                self._resubmit(request_id, state)
                return None, 0
            raise SchedulerError("an admitted request lost its KV completion guarantee") from exc

        return (
            ScheduledRequest(
                request_id=request_id,
                num_computed_tokens=state.num_computed_tokens,
                num_scheduled_tokens=num_scheduled_tokens,
                num_lookahead_tokens=num_lookahead_tokens,
                max_output_tokens=max_output_tokens,
                block_ids=reservation.block_ids,
                num_readonly_prefix_blocks=reservation.num_readonly_prefix_blocks,
            ),
            num_reserved_tokens,
        )

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

        next_num_computed_tokens = state.num_computed_tokens + num_committed_tokens
        next_num_tokens = state.num_tokens + num_new_tokens
        if next_num_tokens > state.max_num_tokens:
            raise SchedulerError(f"request {request_id!r} exceeded its token limit")
        if next_num_computed_tokens > next_num_tokens:
            raise SchedulerError(f"request {request_id!r} committed unknown tokens")

        # 提交已算完的输入，以及投机解码中已经写入缓存的新 token；其余预留释放。
        self._kv_cache.commit(request_id, num_committed_tokens)
        state.num_computed_tokens = next_num_computed_tokens
        # 尚未写入缓存的新 token 会在下一轮作为普通输入继续计算。
        state.num_tokens = next_num_tokens
        had_visible_output = state.has_visible_output
        state.has_visible_output = had_visible_output or num_new_tokens > 0
        if state.is_short and not had_visible_output and state.has_visible_output:
            self._short_launch_order.remove(request_id)
            self._common_order.append(request_id)

    def _classify_waiting_requests(self) -> tuple[tuple[str, bool], ...]:
        return tuple(
            (request_id, self._is_short_request(self._states[request_id]))
            for request_id in self._waiting
        )

    def _age_waiting_requests(
        self,
        classifications: tuple[tuple[str, bool], ...],
    ) -> None:
        if self._short_request_policy is not None:
            for request_id, is_short in classifications:
                if is_short:
                    continue
                state = self._states[request_id]
                state.waiting_steps += 1

    def _admit_waiting_requests(
        self,
        classifications: tuple[tuple[str, bool], ...],
    ) -> None:
        if not self._waiting:
            return
        policy = self._short_request_policy
        if policy is None:
            while self._waiting and len(self._running) < self._max_num_sequences:
                if not self._try_admit(self._waiting[0], is_short=False, min_free_tokens=0):
                    return
            return

        barrier = next(
            (
                request_id
                for request_id, is_short in classifications
                if not is_short
                and self._states[request_id].waiting_steps >= policy.regular_aging_steps
            ),
            None,
        )
        # 老化的常规请求暂时不再为后来短请求让出 KV 水位。
        if barrier is not None:
            if not self._has_common_admission_slot():
                return
            if not self._try_admit(
                barrier,
                is_short=False,
                min_free_tokens=0,
            ):
                return

        for desired_short, min_free_tokens in (
            (True, 0),
            (False, policy.reserved_kv_token_slots),
        ):
            candidates = tuple(
                request_id
                for request_id, is_short in classifications
                if is_short is desired_short and request_id in self._waiting
            )
            for request_id in candidates:
                has_slot = (
                    self._has_short_admission_slot()
                    if desired_short
                    else self._has_common_admission_slot()
                )
                if not has_slot:
                    break
                if not self._try_admit(
                    request_id,
                    is_short=desired_short,
                    min_free_tokens=min_free_tokens,
                ):
                    # lane 内保持 FCFS，不让后来的小请求反复绕过队首。
                    break

        if not self._running and self._waiting:
            # 没有可运行请求时不空转；最老请求可临时借用短请求水位。
            oldest = self._waiting[0]
            is_short = next(
                is_short for request_id, is_short in classifications if request_id == oldest
            )
            self._try_admit(oldest, is_short=is_short, min_free_tokens=0)

    def _try_admit(
        self,
        request_id: str,
        *,
        is_short: bool,
        min_free_tokens: int,
    ) -> bool:
        state = self._states[request_id]
        guarantee_completion = (
            self._self_resubmit_policy is None or is_short or state.force_completion_claim
        )
        match = self._kv_cache.try_add_request(
            request_id,
            token_ids=state.prompt_token_ids,
            # 最后一个可见输出用于结束请求，不再需要写入 KV。
            max_num_committed_tokens=state.max_num_tokens - 1,
            cache_epoch=state.cache_epoch,
            min_free_token_slots=min_free_tokens,
            guarantee_completion=guarantee_completion,
        )
        if match is None:
            return False
        if not 0 <= match.num_cached_tokens < len(state.prompt_token_ids):
            self._kv_cache.free(request_id)
            raise SchedulerError("cached prefix must leave at least one token to compute")

        self._waiting.remove(request_id)
        state.num_computed_tokens = match.num_cached_tokens
        state.kv_attached = True
        state.is_short = is_short
        state.uses_short_admission_slot = is_short
        state.has_completion_claim = guarantee_completion
        self._running[request_id] = state
        order = self._short_launch_order if is_short else self._common_order
        order.append(request_id)
        return True

    def _is_short_request(self, state: _RequestState) -> bool:
        policy = self._short_request_policy
        if (
            policy is None
            or state.has_visible_output
            or state.max_num_tokens > policy.max_total_tokens
        ):
            return False
        match = self._kv_cache.preview_prefix(
            token_ids=state.prompt_token_ids,
            cache_epoch=state.cache_epoch,
        )
        effective_prompt_tokens = len(state.prompt_token_ids) - match.num_cached_tokens
        return effective_prompt_tokens <= policy.max_effective_prompt_tokens

    def _resubmit(self, request_id: str, state: _RequestState) -> None:
        """释放撞墙者自己的 KV，并把完整 token 历史留给 Engine 重算。"""

        self._running.pop(request_id)
        for order in (self._short_launch_order, self._common_order):
            with suppress(ValueError):
                order.remove(request_id)
        self._kv_cache.free(request_id)

        state.num_resubmits += 1
        state.num_recomputed_tokens += state.num_computed_tokens
        self._self_resubmits_total += 1
        self._self_resubmit_recomputed_tokens_total += state.num_computed_tokens
        state.num_computed_tokens = 0
        state.kv_attached = False
        state.is_short = False
        state.uses_short_admission_slot = False
        state.has_completion_claim = False

        policy = self._self_resubmit_policy
        assert policy is not None
        if (
            state.num_resubmits >= policy.max_resubmits
            or state.num_recomputed_tokens >= policy.strict_fallback_recomputed_tokens
        ):
            state.force_completion_claim = True
        self._waiting.append(request_id)
        self._resubmitted_in_step.append(request_id)

    def _rebalance_admission_slots(self) -> None:
        """有通用 slot 时，把已出首 token 的短请求迁出准入保留区。"""

        if self._short_request_policy is None:
            return
        for request_id in self._common_order:
            if not self._has_common_admission_slot():
                return
            state = self._running[request_id]
            if state.uses_short_admission_slot:
                state.uses_short_admission_slot = False

    def _has_short_admission_slot(self) -> bool:
        policy = self._short_request_policy
        assert policy is not None
        used = sum(state.uses_short_admission_slot for state in self._running.values())
        return used < policy.reserved_sequences

    def _has_common_admission_slot(self) -> bool:
        policy = self._short_request_policy
        assert policy is not None
        short_slots = sum(state.uses_short_admission_slot for state in self._running.values())
        common_slots = len(self._running) - short_slots
        return common_slots < self._max_num_sequences - policy.reserved_sequences

    @staticmethod
    def _rotate_request_ids(order: deque[str]) -> tuple[str, ...]:
        request_ids = tuple(order)
        if request_ids:
            order.rotate(-1)
        return request_ids

    @staticmethod
    def _pending_tokens(state: _RequestState) -> int:
        return state.num_tokens - state.num_computed_tokens

    @staticmethod
    def _max_remaining_tokens(state: _RequestState) -> int:
        return state.max_num_tokens - state.num_computed_tokens
