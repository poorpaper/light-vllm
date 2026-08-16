import pytest
import torch

from light_vllm.runtime.execution import (
    ExecutionBatch,
    ExecutionRequest,
    GreedyAcceptanceSampler,
    NGramSpeculativeDecodeHandler,
    NGramTokenProposer,
)
from light_vllm.runtime.sampling import GreedySampler


def test_ngram_proposer_prefers_the_longest_recent_match() -> None:
    proposer = NGramTokenProposer(min_match_length=2, max_match_length=4)

    # 最后的 1,2,3 在较早位置出现过，后面的 8,9 可以直接作为候选。
    proposal = proposer.propose((1, 2, 3, 8, 9, 1, 2, 3), max_tokens=3)

    assert proposal == (8, 9, 1)


def test_ngram_proposer_uses_the_most_recent_match_and_respects_the_limit() -> None:
    proposer = NGramTokenProposer(min_match_length=2, max_match_length=2)

    proposal = proposer.propose((1, 2, 7, 1, 2, 8, 1, 2), max_tokens=1)

    assert proposal == (8,)


def test_ngram_proposer_returns_empty_without_a_repeated_suffix() -> None:
    proposer = NGramTokenProposer()

    assert proposer.propose((1, 2, 3, 4), max_tokens=4) == ()


def test_ngram_proposer_supports_overlapping_history_matches() -> None:
    proposer = NGramTokenProposer(min_match_length=2, max_match_length=2)

    assert proposer.propose((1, 2, 1, 2), max_tokens=2) == (1, 2)


@pytest.mark.parametrize(
    ("drafts", "targets", "expected_outputs", "expected_cached"),
    [
        ((2, 3), (2, 3, 4), (2, 3, 4), 2),
        ((2, 9, 8), (2, 3, 4, 5), (2, 3), 1),
        ((9, 8), (2, 3, 4), (2,), 0),
        ((), (2,), (2,), 0),
    ],
)
def test_greedy_acceptance_returns_confirmed_tokens_and_cached_prefix(
    drafts: tuple[int, ...],
    targets: tuple[int, ...],
    expected_outputs: tuple[int, ...],
    expected_cached: int,
) -> None:
    result = GreedyAcceptanceSampler().accept(drafts, targets)

    assert result.output_token_ids == expected_outputs
    assert result.num_cached_output_tokens == expected_cached


def test_greedy_acceptance_requires_one_more_target_than_drafts() -> None:
    with pytest.raises(ValueError, match="one prediction"):
        GreedyAcceptanceSampler().accept((1,), (1,))


class _FixedProposer:
    def __init__(self, drafts: tuple[int, ...]) -> None:
        self._drafts = drafts

    def propose(self, token_ids: tuple[int, ...], *, max_tokens: int) -> tuple[int, ...]:
        return self._drafts[:max_tokens]


class _TargetStep:
    def __init__(self, target_token_ids: tuple[int, ...], *, original_input_length: int) -> None:
        self._target_token_ids = target_token_ids
        self._original_input_length = original_input_length
        self.verification_batch: ExecutionBatch | None = None
        self.truncated_to: tuple[str, int] | None = None

    def forward(self, model, batch: ExecutionBatch) -> tuple[torch.Tensor, ...]:
        self.verification_batch = batch
        request = batch.requests[0]
        logits = torch.zeros((len(request.input_token_ids), 16))
        first_target_row = self._original_input_length - 1
        for offset, token_id in enumerate(self._target_token_ids):
            logits[first_target_row + offset, token_id] = 1
        return (logits,)

    def truncate(self, request_id: str, num_cached_tokens: int) -> None:
        self.truncated_to = (request_id, num_cached_tokens)


@pytest.mark.parametrize(
    ("drafts", "targets", "expected_outputs", "expected_cached"),
    [
        ((6, 7), (6, 7, 8), (6, 7, 8), 2),
        ((6, 7), (6, 9, 8), (6, 9), 1),
        ((6, 7), (9, 7, 8), (9,), 0),
        ((), (8,), (8,), 0),
    ],
)
def test_speculative_handler_verifies_and_truncates_the_draft_tail(
    drafts: tuple[int, ...],
    targets: tuple[int, ...],
    expected_outputs: tuple[int, ...],
    expected_cached: int,
) -> None:
    handler = NGramSpeculativeDecodeHandler(
        _FixedProposer(drafts),
        GreedySampler(),
        GreedyAcceptanceSampler(),
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

    result = handler.execute(object(), ExecutionBatch(requests=(request,)), step).requests[0]

    assert result.output_token_ids == expected_outputs
    assert result.num_cached_output_tokens == expected_cached
    assert step.truncated_to == ("request", 4 + expected_cached)
    assert step.verification_batch is not None
    verification = step.verification_batch.requests[0]
    assert verification.input_token_ids == (4, 5) + drafts
    assert verification.context_token_ids == (1, 2, 4, 5) + drafts
    assert verification.num_lookahead_tokens == 2 - len(drafts)
