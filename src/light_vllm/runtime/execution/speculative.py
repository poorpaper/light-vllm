"""可替换的投机候选与验收策略。"""

from __future__ import annotations

from light_vllm.runtime.execution.interfaces import (
    AcceptanceResult,
    AcceptanceSampler,
    ExecutionBatch,
    ExecutionError,
    ExecutionOutput,
    ExecutionRequest,
    ModelStepHandler,
    RequestOutput,
    TokenProposer,
)
from light_vllm.runtime.sampling import Sampler


class NGramTokenProposer:
    """从当前请求的历史中续写最近出现过的重复片段。

    先找最长的重复后缀；长度相同时选离当前位置最近的一次。这个实现没有
    额外模型，也不保存跨请求状态，适合作为清晰、稳定的投机解码起点。
    """

    def __init__(self, *, min_match_length: int = 2, max_match_length: int = 5) -> None:
        if type(min_match_length) is not int or min_match_length <= 0:
            raise ValueError("min_match_length must be a positive integer")
        if type(max_match_length) is not int or max_match_length < min_match_length:
            raise ValueError("max_match_length must be at least min_match_length")
        self._min_match_length = min_match_length
        self._max_match_length = max_match_length

    def propose(self, token_ids: tuple[int, ...], *, max_tokens: int) -> tuple[int, ...]:
        token_ids = tuple(token_ids)
        if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
            raise ValueError("token_ids must contain non-negative integers")
        if type(max_tokens) is not int or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        if max_tokens == 0:
            return ()

        max_length = min(self._max_match_length, len(token_ids) - 1)
        for match_length in range(max_length, self._min_match_length - 1, -1):
            suffix = token_ids[-match_length:]
            # 当前后缀本身不能当作命中；倒序查找可直接得到最近一次。
            for start in range(len(token_ids) - match_length - 1, -1, -1):
                if token_ids[start : start + match_length] == suffix:
                    proposal_start = start + match_length
                    return token_ids[proposal_start : proposal_start + max_tokens]
        return ()


class GreedyAcceptanceSampler:
    """候选与目标结果相同就继续接受，第一次不同处改用目标 token。"""

    def accept(
        self,
        draft_token_ids: tuple[int, ...],
        target_token_ids: tuple[int, ...],
    ) -> AcceptanceResult:
        draft_token_ids = tuple(draft_token_ids)
        target_token_ids = tuple(target_token_ids)
        if any(
            type(token_id) is not int or token_id < 0
            for token_id in draft_token_ids + target_token_ids
        ):
            raise ValueError("draft and target tokens must be non-negative integers")
        if len(target_token_ids) != len(draft_token_ids) + 1:
            raise ValueError("target tokens must contain one prediction after every draft")

        for index, draft_token_id in enumerate(draft_token_ids):
            if draft_token_id != target_token_ids[index]:
                return AcceptanceResult(
                    output_token_ids=draft_token_ids[:index] + (target_token_ids[index],),
                    num_cached_output_tokens=index,
                )
        return AcceptanceResult(
            output_token_ids=draft_token_ids + (target_token_ids[-1],),
            num_cached_output_tokens=len(draft_token_ids),
        )


class NGramSpeculativeDecodeHandler:
    """用历史候选扩展本轮输入，再由目标模型一次验证。"""

    def __init__(
        self,
        proposer: TokenProposer,
        target_sampler: Sampler,
        acceptance_sampler: AcceptanceSampler,
    ) -> None:
        self._proposer = proposer
        self._target_sampler = target_sampler
        self._acceptance_sampler = acceptance_sampler

    def execute(self, model, batch: ExecutionBatch, step: ModelStepHandler) -> ExecutionOutput:
        drafts_by_request: list[tuple[int, ...]] = []
        verification_requests: list[ExecutionRequest] = []
        for request in batch.requests:
            max_drafts = min(
                request.num_lookahead_tokens,
                max(0, request.max_output_tokens - 1),
            )
            drafts = (
                tuple(
                    self._proposer.propose(
                        request.context_token_ids,
                        max_tokens=max_drafts,
                    )
                )
                if max_drafts
                else ()
            )
            if len(drafts) > max_drafts:
                raise ExecutionError("token proposer returned more tokens than requested")
            if any(type(token_id) is not int or token_id < 0 for token_id in drafts):
                raise ExecutionError("token proposer returned an invalid token ID")
            drafts_by_request.append(drafts)
            verification_requests.append(
                ExecutionRequest(
                    request_id=request.request_id,
                    input_token_ids=request.input_token_ids + drafts,
                    context_token_ids=request.context_token_ids + drafts,
                    num_computed_tokens=request.num_computed_tokens,
                    # 候选不足时保留剩余预留事实，分页表仍能做严格校验。
                    num_lookahead_tokens=request.num_lookahead_tokens - len(drafts),
                    max_output_tokens=request.max_output_tokens,
                    block_ids=request.block_ids,
                    num_readonly_prefix_blocks=request.num_readonly_prefix_blocks,
                )
            )

        logits_by_request = step.forward(
            model,
            ExecutionBatch(requests=tuple(verification_requests)),
        )
        if len(logits_by_request) != len(batch.requests):
            raise ExecutionError("model step must return one logits tensor per request")

        results: list[RequestOutput] = []
        for request, verification, drafts, logits in zip(
            batch.requests,
            verification_requests,
            drafts_by_request,
            logits_by_request,
            strict=True,
        ):
            if logits.ndim != 2 or logits.shape[0] != len(verification.input_token_ids):
                raise ExecutionError("model step logits must have shape [query, vocabulary]")
            if not request.max_output_tokens:
                results.append(
                    RequestOutput(
                        request_id=request.request_id,
                        num_input_tokens_computed=len(request.input_token_ids),
                    )
                )
                continue

            # 原输入最后一行预测第一个候选；随后每行依次预测下一个 token。
            first_target_row = len(request.input_token_ids) - 1
            target_logits = logits[first_target_row : first_target_row + len(drafts) + 1]
            target_token_ids = self._target_sampler.sample(target_logits)
            if len(target_token_ids) != len(drafts) + 1:
                raise ExecutionError("target sampler returned the wrong number of tokens")
            accepted = self._acceptance_sampler.accept(drafts, target_token_ids)
            if not isinstance(accepted, AcceptanceResult):
                raise ExecutionError("acceptance sampler must return AcceptanceResult")
            if len(accepted.output_token_ids) > request.max_output_tokens:
                raise ExecutionError("acceptance sampler exceeded the output budget")
            if accepted.num_cached_output_tokens > len(drafts):
                raise ExecutionError("acceptance sampler cached unverified tokens")

            # 目标 forward 已写入全部候选，只保留真正接受的候选前缀。
            step.truncate(
                request.request_id,
                request.num_computed_tokens
                + len(request.input_token_ids)
                + accepted.num_cached_output_tokens,
            )
            results.append(
                RequestOutput(
                    request_id=request.request_id,
                    num_input_tokens_computed=len(request.input_token_ids),
                    output_token_ids=accepted.output_token_ids,
                    num_cached_output_tokens=accepted.num_cached_output_tokens,
                )
            )
        return ExecutionOutput(
            requests=tuple(results),
            num_model_tokens_computed=sum(
                len(request.input_token_ids) for request in verification_requests
            ),
        )
