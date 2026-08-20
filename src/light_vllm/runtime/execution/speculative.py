"""可替换的草稿树、目标验收与投机解码流程。"""

from __future__ import annotations

from dataclasses import dataclass, field

from light_vllm.runtime.execution.interfaces import (
    AcceptanceResult,
    AcceptanceSampler,
    DraftProposer,
    DraftTree,
    ExecutionBatch,
    ExecutionError,
    ExecutionOutput,
    ModelStepBatch,
    ModelStepHandler,
    ModelStepRequest,
    RequestOutput,
    SpeculationObserver,
    SpeculativeDecodeObservation,
)
from light_vllm.runtime.execution.layout import append_draft_layout
from light_vllm.runtime.sampling import Sampler


def _validate_ngram_config(min_match_length: int, max_match_length: int) -> None:
    if type(min_match_length) is not int or min_match_length <= 0:
        raise ValueError("min_match_length must be a positive integer")
    if type(max_match_length) is not int or max_match_length < min_match_length:
        raise ValueError("max_match_length must be at least min_match_length")


def _validate_proposal_input(token_ids: tuple[int, ...], max_nodes: int) -> None:
    if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
        raise ValueError("token_ids must contain non-negative integers")
    if type(max_nodes) is not int or max_nodes < 0:
        raise ValueError("max_nodes must be a non-negative integer")


def _matching_starts(
    token_ids: tuple[int, ...],
    *,
    min_match_length: int,
    max_match_length: int,
) -> tuple[int, tuple[int, ...]]:
    """找到最长重复后缀及其全部历史起点。"""

    max_length = min(max_match_length, len(token_ids) - 1)
    for match_length in range(max_length, min_match_length - 1, -1):
        suffix = token_ids[-match_length:]
        starts = tuple(
            start
            for start in range(len(token_ids) - match_length)
            if token_ids[start : start + match_length] == suffix
        )
        if starts:
            return match_length, starts
    return 0, ()


class NGramChainProposer:
    """把最近一次最长 N-Gram 命中转换成线性草稿树。"""

    def __init__(self, *, min_match_length: int = 2, max_match_length: int = 5) -> None:
        _validate_ngram_config(min_match_length, max_match_length)
        self._min_match_length = min_match_length
        self._max_match_length = max_match_length

    def propose(self, token_ids: tuple[int, ...], *, max_nodes: int) -> DraftTree:
        token_ids = tuple(token_ids)
        _validate_proposal_input(token_ids, max_nodes)
        if max_nodes == 0:
            return DraftTree((), ())

        match_length, starts = _matching_starts(
            token_ids,
            min_match_length=self._min_match_length,
            max_match_length=self._max_match_length,
        )
        if not starts:
            return DraftTree((), ())
        proposal_start = starts[-1] + match_length
        drafts = token_ids[proposal_start : proposal_start + max_nodes]
        return DraftTree(
            token_ids=drafts,
            parent_indices=tuple(-1 if index == 0 else index - 1 for index in range(len(drafts))),
        )


@dataclass(slots=True)
class _TrieNode:
    token_id: int
    frequency: int = 0
    last_occurrence: int = -1
    children: dict[int, _TrieNode] = field(default_factory=dict)

    def observe(self, occurrence: int) -> None:
        self.frequency += 1
        self.last_occurrence = max(self.last_occurrence, occurrence)


class NGramTrieProposer:
    """把最长 N-Gram 的全部历史续写整理成有界、确定性的 Trie。"""

    def __init__(
        self,
        *,
        min_match_length: int = 2,
        max_match_length: int = 5,
        max_depth: int = 4,
        max_branching: int = 4,
    ) -> None:
        _validate_ngram_config(min_match_length, max_match_length)
        if type(max_depth) is not int or max_depth <= 0:
            raise ValueError("max_depth must be a positive integer")
        if type(max_branching) is not int or max_branching <= 0:
            raise ValueError("max_branching must be a positive integer")
        self._min_match_length = min_match_length
        self._max_match_length = max_match_length
        self._max_depth = max_depth
        self._max_branching = max_branching

    def propose(self, token_ids: tuple[int, ...], *, max_nodes: int) -> DraftTree:
        token_ids = tuple(token_ids)
        _validate_proposal_input(token_ids, max_nodes)
        if max_nodes == 0:
            return DraftTree((), ())

        match_length, starts = _matching_starts(
            token_ids,
            min_match_length=self._min_match_length,
            max_match_length=self._max_match_length,
        )
        if not starts:
            return DraftTree((), ())

        roots: dict[int, _TrieNode] = {}
        for start in starts:
            continuation_start = start + match_length
            continuation = token_ids[continuation_start : continuation_start + self._max_depth]
            children = roots
            for token_id in continuation:
                node = children.setdefault(token_id, _TrieNode(token_id))
                node.observe(start)
                children = node.children

        def ranked(children: dict[int, _TrieNode]) -> tuple[_TrieNode, ...]:
            return tuple(
                sorted(
                    children.values(),
                    key=lambda node: (
                        -node.frequency,
                        -node.last_occurrence,
                        node.token_id,
                    ),
                )[: self._max_branching]
            )

        token_ids_by_node: list[int] = []
        parent_indices: list[int] = []
        frontier: list[tuple[_TrieNode, int, int]] = []

        # First retain root alternatives: a trie is useful only if it can rescue
        # the recent-chain miss at the first token. Then spend the remaining
        # global budget best-first, preserving depth on the historically most
        # likely path instead of breadth-first diluting every branch equally.
        for node in ranked(roots):
            if len(token_ids_by_node) >= max_nodes:
                break
            node_index = len(token_ids_by_node)
            token_ids_by_node.append(node.token_id)
            parent_indices.append(-1)
            if self._max_depth > 1:
                frontier.extend((child, node_index, 2) for child in ranked(node.children))

        while frontier and len(token_ids_by_node) < max_nodes:
            best = min(
                range(len(frontier)),
                key=lambda index: (
                    -frontier[index][0].frequency,
                    -frontier[index][0].last_occurrence,
                    frontier[index][0].token_id,
                    frontier[index][1],
                ),
            )
            node, parent, depth = frontier.pop(best)
            node_index = len(token_ids_by_node)
            token_ids_by_node.append(node.token_id)
            parent_indices.append(parent)
            if depth < self._max_depth:
                frontier.extend((child, node_index, depth + 1) for child in ranked(node.children))

        # Flatten the selected tree depth-first. The most likely accepted path
        # stays physically contiguous, so the paged backend does not have to
        # compact K/V merely because alternative roots consumed early slots.
        children_by_parent: dict[int, list[int]] = {}
        for index, parent in enumerate(parent_indices):
            children_by_parent.setdefault(parent, []).append(index)
        depth_first_order: list[int] = []

        def append_subtree(index: int) -> None:
            depth_first_order.append(index)
            for child in children_by_parent.get(index, ()):
                append_subtree(child)

        for root in children_by_parent.get(-1, ()):
            append_subtree(root)
        remapped_indices = {
            old_index: new_index for new_index, old_index in enumerate(depth_first_order)
        }
        return DraftTree(
            tuple(token_ids_by_node[index] for index in depth_first_order),
            tuple(
                -1 if parent_indices[index] == -1 else remapped_indices[parent_indices[index]]
                for index in depth_first_order
            ),
        )


class GreedyTreeAcceptanceSampler:
    """沿目标模型的 greedy token 在草稿树中逐层选择唯一子节点。"""

    def accept(
        self,
        draft: DraftTree,
        target_token_ids: tuple[int, ...],
    ) -> AcceptanceResult:
        target_token_ids = tuple(target_token_ids)
        if any(type(token_id) is not int or token_id < 0 for token_id in target_token_ids):
            raise ValueError("target tokens must be non-negative integers")
        if len(target_token_ids) != len(draft) + 1:
            raise ValueError("target tokens must contain one prediction after every draft")

        child_by_token = {
            (parent, token_id): index
            for index, (token_id, parent) in enumerate(
                zip(draft.token_ids, draft.parent_indices, strict=True)
            )
        }
        output_token_ids: list[int] = []
        accepted_indices: list[int] = []
        parent = -1
        target_index = 0
        while True:
            target_token_id = target_token_ids[target_index]
            child = child_by_token.get((parent, target_token_id))
            output_token_ids.append(target_token_id)
            if child is None:
                break
            accepted_indices.append(child)
            parent = child
            target_index = child + 1

        return AcceptanceResult(
            output_token_ids=tuple(output_token_ids),
            accepted_draft_indices=tuple(accepted_indices),
        )


def _validate_accepted_path(accepted: AcceptanceResult, draft: DraftTree) -> None:
    parent = -1
    for output_offset, draft_index in enumerate(accepted.accepted_draft_indices):
        if draft_index >= len(draft):
            raise ExecutionError("acceptance sampler returned an unknown draft index")
        if draft.parent_indices[draft_index] != parent:
            raise ExecutionError("acceptance sampler returned indices outside one draft path")
        if accepted.output_token_ids[output_offset] != draft.token_ids[draft_index]:
            raise ExecutionError("acceptance sampler returned a token outside the accepted path")
        parent = draft_index


def _draft_shape(draft: DraftTree) -> tuple[int, int, int]:
    """返回根节点数、真实分支父节点数和最大深度。"""

    depths: list[int] = []
    for parent in draft.parent_indices:
        depths.append(1 if parent == -1 else depths[parent] + 1)
    roots = sum(parent == -1 for parent in draft.parent_indices)
    child_counts: dict[int, int] = {}
    for parent in draft.parent_indices:
        if parent >= 0:
            child_counts[parent] = child_counts.get(parent, 0) + 1
    branching_parents = sum(count > 1 for count in child_counts.values())
    return roots, branching_parents, max(depths, default=0)


class SpeculativeDecodeHandler:
    """把任意草稿树放入一次目标模型 forward，并压实最终命中路径。"""

    def __init__(
        self,
        proposer: DraftProposer,
        target_sampler: Sampler,
        acceptance_sampler: AcceptanceSampler,
        speculation_observer: SpeculationObserver | None = None,
    ) -> None:
        self._proposer = proposer
        self._target_sampler = target_sampler
        self._acceptance_sampler = acceptance_sampler
        self._speculation_observer = speculation_observer

    def _observe(self, observation: SpeculativeDecodeObservation) -> None:
        observer = self._speculation_observer
        if observer is None:
            return
        try:
            observer.speculation_completed(observation)
        except Exception:
            # 指标是旁路；首次失败后停用，不能改变生成结果或持续增加热路径开销。
            self._speculation_observer = None

    def execute(self, model, batch: ExecutionBatch, step: ModelStepHandler) -> ExecutionOutput:
        drafts_by_request: list[DraftTree] = []
        model_requests: list[ModelStepRequest] = []
        for request in batch.requests:
            max_nodes = min(
                request.num_lookahead_tokens,
                max(0, request.max_output_tokens - 1),
            )
            if max_nodes:
                if request.context_token_ids is None:
                    raise ExecutionError("draft proposal requires the complete token context")
                draft = self._proposer.propose(
                    request.context_token_ids,
                    max_nodes=max_nodes,
                )
            else:
                draft = DraftTree((), ())
            if not isinstance(draft, DraftTree):
                raise ExecutionError("draft proposer must return DraftTree")
            if len(draft) > max_nodes:
                raise ExecutionError("draft proposer returned more nodes than requested")
            drafts_by_request.append(draft)
            model_requests.append(
                ModelStepRequest(
                    request_id=request.request_id,
                    query_token_ids=request.input_token_ids + draft.token_ids,
                    num_computed_tokens=request.num_computed_tokens,
                    num_reserved_query_tokens=(
                        len(request.input_token_ids) + request.num_lookahead_tokens
                    ),
                    query_layout=append_draft_layout(
                        len(request.input_token_ids),
                        draft,
                    ),
                    block_ids=request.block_ids,
                    num_readonly_prefix_blocks=request.num_readonly_prefix_blocks,
                )
            )

        model_batch = ModelStepBatch(tuple(model_requests))
        logits_by_request = step.forward(model, model_batch)
        if len(logits_by_request) != len(batch.requests):
            raise ExecutionError("model step must return one logits tensor per request")

        results: list[RequestOutput] = []
        observations: list[SpeculativeDecodeObservation] = []
        for request, model_request, draft, logits in zip(
            batch.requests,
            model_requests,
            drafts_by_request,
            logits_by_request,
            strict=True,
        ):
            if logits.ndim != 2 or logits.shape[0] != len(model_request.query_token_ids):
                raise ExecutionError("model step logits must have shape [query, vocabulary]")
            if not request.max_output_tokens:
                results.append(
                    RequestOutput(
                        request_id=request.request_id,
                        num_input_tokens_computed=len(request.input_token_ids),
                    )
                )
                continue

            # 正式输入最后一行预测草稿根；每个草稿节点行预测自己的子节点。
            first_target_row = len(request.input_token_ids) - 1
            target_logits = logits[first_target_row : first_target_row + len(draft) + 1]
            target_token_ids = self._target_sampler.sample(target_logits)
            if len(target_token_ids) != len(draft) + 1:
                raise ExecutionError("target sampler returned the wrong number of tokens")
            accepted = self._acceptance_sampler.accept(draft, target_token_ids)
            if not isinstance(accepted, AcceptanceResult):
                raise ExecutionError("acceptance sampler must return AcceptanceResult")
            _validate_accepted_path(accepted, draft)
            if len(accepted.output_token_ids) > request.max_output_tokens:
                raise ExecutionError("acceptance sampler exceeded the output budget")

            retained_query_indices = tuple(range(len(request.input_token_ids))) + tuple(
                len(request.input_token_ids) + index for index in accepted.accepted_draft_indices
            )
            num_compacted_tokens = 0
            if len(draft):
                num_compacted_tokens = step.compact(
                    model_request,
                    retained_query_indices,
                )
                if type(num_compacted_tokens) is not int or not 0 <= num_compacted_tokens <= len(
                    accepted.accepted_draft_indices
                ):
                    raise ExecutionError("model step returned an invalid compacted-token count")
            if request.num_lookahead_tokens:
                roots, branching_parents, max_depth = _draft_shape(draft)
                observations.append(
                    SpeculativeDecodeObservation(
                        num_proposed_nodes=len(draft),
                        num_accepted_nodes=len(accepted.accepted_draft_indices),
                        num_verified_tokens=len(accepted.output_token_ids),
                        num_draft_roots=roots,
                        num_branching_parents=branching_parents,
                        max_draft_depth=max_depth,
                        num_compacted_tokens=num_compacted_tokens,
                    )
                )
            results.append(
                RequestOutput(
                    request_id=request.request_id,
                    num_input_tokens_computed=len(request.input_token_ids),
                    output_token_ids=accepted.output_token_ids,
                    num_cached_output_tokens=accepted.num_cached_output_tokens,
                )
            )
        # 整个 batch 都通过输出与 compact 校验后，才发布旁路统计。
        for observation in observations:
            self._observe(observation)
        return ExecutionOutput(
            requests=tuple(results),
            num_model_tokens_computed=sum(
                len(request.query_token_ids) for request in model_requests
            ),
        )
