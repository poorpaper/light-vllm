import pytest

from light_vllm.runtime.execution import GreedyAcceptanceSampler, NGramTokenProposer


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
