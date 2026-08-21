import pytest
import torch

from light_vllm.runtime.execution import (
    DraftTree,
    ExecutionBatch,
    ExecutionError,
    ExecutionRequest,
    GreedyTreeAcceptanceSampler,
    ModelStepBatch,
    ModelStepOutput,
    NGramChainProposer,
    NGramTrieProposer,
    SpeculativeDecodeHandler,
    SpeculativeDecodeObservation,
)
from light_vllm.runtime.sampling import GreedySampler


def test_ngram_chain_prefers_the_longest_recent_match() -> None:
    proposer = NGramChainProposer(min_match_length=2, max_match_length=4)

    proposal = proposer.propose((1, 2, 3, 8, 9, 1, 2, 3), max_nodes=3)

    assert proposal == DraftTree((8, 9, 1), (-1, 0, 1))


def test_ngram_chain_uses_the_most_recent_match_and_respects_the_limit() -> None:
    proposer = NGramChainProposer(min_match_length=2, max_match_length=2)

    proposal = proposer.propose((1, 2, 7, 1, 2, 8, 1, 2), max_nodes=1)

    assert proposal == DraftTree((8,), (-1,))


def test_ngram_chain_returns_empty_without_a_repeated_suffix() -> None:
    proposer = NGramChainProposer()

    assert proposer.propose((1, 2, 3, 4), max_nodes=4) == DraftTree((), ())


def test_ngram_chain_supports_overlapping_history_matches() -> None:
    proposer = NGramChainProposer(min_match_length=2, max_match_length=2)

    assert proposer.propose((1, 2, 1, 2), max_nodes=2) == DraftTree(
        (1, 2),
        (-1, 0),
    )


def test_ngram_trie_collects_branches_in_deterministic_order() -> None:
    proposer = NGramTrieProposer(
        min_match_length=2,
        max_match_length=2,
        max_depth=2,
        max_branching=2,
    )

    proposal = proposer.propose(
        (1, 2, 7, 8, 1, 2, 7, 9, 1, 2),
        max_nodes=4,
    )

    # 7 出现两次；同频子节点优先选择更近的 9，再选择 8。
    assert proposal == DraftTree((7, 9, 8), (-1, 0, 0))


def test_ngram_trie_covers_roots_then_preserves_the_frequent_path_depth() -> None:
    proposer = NGramTrieProposer(
        min_match_length=2,
        max_match_length=2,
        max_depth=3,
        max_branching=2,
    )

    proposal = proposer.propose(
        (1, 2, 7, 8, 9, 1, 2, 7, 8, 9, 1, 2, 6, 5, 1, 2),
        max_nodes=4,
    )

    assert proposal == DraftTree((7, 8, 9, 6), (-1, 0, 1, -1))


def test_ngram_trie_respects_global_node_budget() -> None:
    proposer = NGramTrieProposer(
        min_match_length=2,
        max_match_length=2,
        max_depth=3,
        max_branching=3,
    )

    proposal = proposer.propose(
        (1, 2, 7, 3, 1, 2, 8, 4, 1, 2),
        max_nodes=1,
    )

    assert len(proposal) == 1


def _linear_draft(token_ids: tuple[int, ...]) -> DraftTree:
    return DraftTree(
        token_ids,
        tuple(-1 if index == 0 else index - 1 for index in range(len(token_ids))),
    )


@pytest.mark.parametrize(
    ("drafts", "targets", "expected_outputs", "expected_cached"),
    [
        ((2, 3), (2, 3, 4), (2, 3, 4), 2),
        ((2, 9, 8), (2, 3, 4, 5), (2, 3), 1),
        ((9, 8), (2, 3, 4), (2,), 0),
        ((), (2,), (2,), 0),
    ],
)
def test_greedy_tree_acceptance_preserves_linear_behavior(
    drafts: tuple[int, ...],
    targets: tuple[int, ...],
    expected_outputs: tuple[int, ...],
    expected_cached: int,
) -> None:
    result = GreedyTreeAcceptanceSampler().accept(_linear_draft(drafts), targets)

    assert result.output_token_ids == expected_outputs
    assert result.num_cached_output_tokens == expected_cached


def test_greedy_tree_acceptance_follows_a_non_contiguous_flattened_path() -> None:
    draft = DraftTree(
        token_ids=(6, 9, 7, 10),
        parent_indices=(-1, -1, 0, 2),
    )

    result = GreedyTreeAcceptanceSampler().accept(
        draft,
        target_token_ids=(6, 7, 0, 10, 11),
    )

    assert result.output_token_ids == (6, 7, 10, 11)
    assert result.accepted_draft_indices == (0, 2, 3)


def test_greedy_tree_acceptance_requires_one_more_target_than_drafts() -> None:
    with pytest.raises(ValueError, match="one prediction"):
        GreedyTreeAcceptanceSampler().accept(_linear_draft((1,)), (1,))


class _FixedProposer:
    def __init__(self, drafts: tuple[int, ...]) -> None:
        self._draft = _linear_draft(drafts)

    def propose(self, token_ids: tuple[int, ...], *, max_nodes: int) -> DraftTree:
        return DraftTree(
            self._draft.token_ids[:max_nodes],
            self._draft.parent_indices[:max_nodes],
        )


class _TargetStep:
    def __init__(self, target_token_ids: tuple[int, ...], *, original_input_length: int) -> None:
        self._target_token_ids = target_token_ids
        self._original_input_length = original_input_length
        self.verification_batch: ModelStepBatch | None = None
        self.compacted: tuple[str, tuple[int, ...]] | None = None

    def forward(self, model, batch: ModelStepBatch) -> ModelStepOutput:
        self.verification_batch = batch
        request = batch.requests[0]
        assert request.logit_query_indices == tuple(
            range(
                self._original_input_length - 1,
                self._original_input_length - 1 + len(self._target_token_ids),
            )
        )
        logits = torch.zeros((len(self._target_token_ids), 16))
        for offset, token_id in enumerate(self._target_token_ids):
            logits[offset, token_id] = 1
        return ModelStepOutput(logits, (0, len(self._target_token_ids)))

    def compact(self, request, retained_query_indices: tuple[int, ...]) -> int:
        self.compacted = (request.request_id, retained_query_indices)
        return 0


@pytest.mark.parametrize(
    ("drafts", "targets", "expected_outputs", "expected_cached"),
    [
        ((6, 7), (6, 7, 8), (6, 7, 8), 2),
        ((6, 7), (6, 9, 8), (6, 9), 1),
        ((6, 7), (9, 7, 8), (9,), 0),
        ((), (8,), (8,), 0),
    ],
)
def test_speculative_handler_verifies_and_compacts_the_draft_path(
    drafts: tuple[int, ...],
    targets: tuple[int, ...],
    expected_outputs: tuple[int, ...],
    expected_cached: int,
) -> None:
    handler = SpeculativeDecodeHandler(
        _FixedProposer(drafts),
        GreedySampler(),
        GreedyTreeAcceptanceSampler(),
    )
    request = ExecutionRequest(
        request_id="request",
        input_token_ids=(4, 5),
        context_token_ids=(1, 2, 4, 5),
        num_computed_tokens=2,
        num_lookahead_tokens=2,
        max_output_tokens=3,
        block_ids=None,
    )
    step = _TargetStep(targets, original_input_length=2)

    output = handler.execute(object(), ExecutionBatch(requests=(request,)), step)
    result = output.requests[0]

    assert result.output_token_ids == expected_outputs
    assert result.num_cached_output_tokens == expected_cached
    if drafts:
        assert step.compacted == (
            "request",
            tuple(range(2 + expected_cached)),
        )
    else:
        assert step.compacted is None
    assert step.verification_batch is not None
    verification = step.verification_batch.requests[0]
    assert verification.query_token_ids == (4, 5) + drafts
    assert verification.num_reserved_query_tokens == 4
    assert verification.query_layout.parent_indices == tuple(
        -1 if index == 0 else index - 1 for index in range(2 + len(drafts))
    )
    assert output.num_model_tokens_computed == len(request.input_token_ids) + len(drafts)


def test_speculative_handler_requires_context_when_it_proposes_drafts() -> None:
    handler = SpeculativeDecodeHandler(
        _FixedProposer((6,)),
        GreedySampler(),
        GreedyTreeAcceptanceSampler(),
    )
    request = ExecutionRequest(
        request_id="request",
        input_token_ids=(5,),
        context_token_ids=None,
        num_computed_tokens=1,
        num_lookahead_tokens=1,
        max_output_tokens=2,
        block_ids=None,
    )

    with pytest.raises(ExecutionError, match="complete token context"):
        handler.execute(
            object(),
            ExecutionBatch((request,)),
            _TargetStep((6, 7), original_input_length=1),
        )


def test_speculation_observer_records_the_actual_tree_shape() -> None:
    class TreeProposer:
        def propose(self, token_ids: tuple[int, ...], *, max_nodes: int) -> DraftTree:
            return DraftTree((7, 9, 8), (-1, 0, 0))

    observations: list[SpeculativeDecodeObservation] = []

    class RecordingObserver:
        def speculation_completed(
            self,
            observation: SpeculativeDecodeObservation,
        ) -> None:
            observations.append(observation)

    handler = SpeculativeDecodeHandler(
        TreeProposer(),
        GreedySampler(),
        GreedyTreeAcceptanceSampler(),
        RecordingObserver(),
    )
    request = ExecutionRequest(
        request_id="request",
        input_token_ids=(5,),
        context_token_ids=(5,),
        num_computed_tokens=0,
        num_lookahead_tokens=3,
        max_output_tokens=4,
        block_ids=None,
    )

    handler.execute(
        object(),
        ExecutionBatch((request,)),
        _TargetStep((7, 9, 10, 11), original_input_length=1),
    )

    assert observations == [
        SpeculativeDecodeObservation(
            num_proposed_nodes=3,
            num_accepted_nodes=2,
            num_verified_tokens=3,
            num_draft_roots=1,
            num_branching_parents=1,
            max_draft_depth=2,
            num_compacted_tokens=0,
        )
    ]


def test_speculation_observer_failure_is_disabled_without_changing_output() -> None:
    class FailingObserver:
        def __init__(self) -> None:
            self.calls = 0

        def speculation_completed(
            self,
            observation: SpeculativeDecodeObservation,
        ) -> None:
            self.calls += 1
            raise RuntimeError("metrics unavailable")

    observer = FailingObserver()
    handler = SpeculativeDecodeHandler(
        _FixedProposer((6,)),
        GreedySampler(),
        GreedyTreeAcceptanceSampler(),
        observer,
    )
    request = ExecutionRequest(
        request_id="request",
        input_token_ids=(5,),
        context_token_ids=(5,),
        num_computed_tokens=0,
        num_lookahead_tokens=1,
        max_output_tokens=2,
        block_ids=None,
    )
    batch = ExecutionBatch((request,))

    first = handler.execute(
        object(),
        batch,
        _TargetStep((6, 7), original_input_length=1),
    )
    second = handler.execute(
        object(),
        batch,
        _TargetStep((6, 7), original_input_length=1),
    )

    assert first.requests[0].output_token_ids == (6, 7)
    assert second.requests[0].output_token_ids == (6, 7)
    assert observer.calls == 1


def test_speculation_observation_counts_the_final_target_token() -> None:
    with pytest.raises(ValueError, match="one target token"):
        SpeculativeDecodeObservation(
            num_proposed_nodes=1,
            num_accepted_nodes=1,
            num_verified_tokens=1,
            num_draft_roots=1,
            num_branching_parents=0,
            max_draft_depth=1,
            num_compacted_tokens=0,
        )
