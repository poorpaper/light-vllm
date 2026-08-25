import pytest
import torch

from light_vllm import ConfigurableSampler, SamplingMetadata, SamplingParams
from light_vllm.runtime.sampling import SamplingError


def _metadata(request_id: str, *, seed: int, position: int = 0, **changes):
    params = {"temperature": 1.0, "seed": seed}
    params.update(changes)
    return SamplingMetadata(
        request_id=request_id,
        params=SamplingParams(**params),
        output_position=position,
    )


def test_temperature_zero_is_greedy() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0]])

    assert ConfigurableSampler().sample(
        logits,
        (SamplingMetadata("request", SamplingParams(seed=7), 0),),
    ) == (1,)


def test_top_k_one_and_small_top_p_keep_the_best_token() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0], [1.0, 3.0, 2.0]])

    assert ConfigurableSampler().sample(
        logits,
        (
            _metadata("top-k", seed=3, top_k=1),
            _metadata("top-p", seed=5, top_p=0.01),
        ),
    ) == (1, 1)


def test_very_small_temperature_remains_numerically_stable() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0]])

    assert ConfigurableSampler().sample(
        logits,
        (_metadata("request", seed=3, temperature=1e-300),),
    ) == (1,)


def test_fixed_seed_is_independent_of_batch_row_order() -> None:
    logits = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    first = _metadata("first", seed=11, position=4)
    second = _metadata("second", seed=29, position=4)
    sampler = ConfigurableSampler()

    forward = sampler.sample(logits, (first, second))
    reverse = sampler.sample(logits, (second, first))

    assert forward == tuple(reversed(reverse))


def test_fixed_seed_sequence_is_stable_when_a_peer_enters_and_leaves_the_batch() -> None:
    request_logits = torch.tensor([[0.1, 0.2, 0.3]])
    peer_logits = torch.tensor([[0.8, 0.1, 0.1]])
    sampler = ConfigurableSampler()

    alone = tuple(
        sampler.sample(request_logits, (_metadata("request", seed=11, position=position),))[0]
        for position in range(4)
    )
    interleaved: list[int] = []
    for position in range(4):
        if position % 2:
            sampled = sampler.sample(
                torch.cat((peer_logits, request_logits)),
                (
                    _metadata("peer", seed=29, position=position),
                    _metadata("request", seed=11, position=position),
                ),
            )
            interleaved.append(sampled[1])
        else:
            interleaved.append(
                sampler.sample(
                    request_logits,
                    (_metadata("request", seed=11, position=position),),
                )[0]
            )

    assert tuple(interleaved) == alone


@pytest.mark.parametrize(
    "changes",
    [
        {"temperature": -0.1},
        {"temperature": float("inf")},
        {"top_k": 0},
        {"top_p": 0.0},
        {"seed": -1},
        {"seed": 2**63},
    ],
)
def test_sampling_params_reject_invalid_values(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SamplingParams(**changes)


def test_sampler_rejects_top_k_larger_than_vocabulary() -> None:
    with pytest.raises(SamplingError, match="vocabulary"):
        ConfigurableSampler().sample(
            torch.ones(1, 2),
            (_metadata("request", seed=1, top_k=3),),
        )
