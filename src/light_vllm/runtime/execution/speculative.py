"""可替换的投机候选与验收策略。"""

from __future__ import annotations

from light_vllm.runtime.execution.interfaces import AcceptanceResult


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
