from pathlib import Path

import pytest
import torch
from torch import nn

from light_vllm import ForwardBatch, ModelOutput, ModelSpec, create_catalog, create_runner
from light_vllm.modeling.attention import AttentionLayerSpec, ModelKVCacheSpec
from light_vllm.modeling.models.interfaces import ModelFactory, select_query_states


def tiny_spec(**changes: object) -> ModelSpec:
    values: dict[str, object] = {
        "architecture": "tiny-causal-lm",
        "loader": "init",
        "model_args": {"vocab_size": 32, "hidden_size": 8},
    }
    values.update(changes)
    return ModelSpec(**values)


def test_minimal_forward_shape() -> None:
    runner = create_runner()
    runner.load(tiny_spec())

    batch = ForwardBatch(
        input_ids=torch.tensor([1, 2, 3, 4, 5, 6]),
        positions=torch.tensor([0, 1, 2, 0, 1, 2]),
        query_start_loc=(0, 3, 6),
    )
    output = runner.open_session().forward(batch)

    assert output.logits.shape == (6, 32)
    torch.testing.assert_close(batch.positions, torch.tensor([0, 1, 2, 0, 1, 2]))
    assert runner.generation == 1


def test_forward_batch_validates_explicit_positions() -> None:
    with pytest.raises(ValueError, match="same shape"):
        ForwardBatch(
            input_ids=torch.tensor([1, 2]),
            positions=torch.tensor([4]),
        )

    batch = ForwardBatch(
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([4, 5]),
    )

    torch.testing.assert_close(batch.positions, torch.tensor([4, 5]))


def test_forward_batch_requires_a_boolean_position_trust_marker() -> None:
    with pytest.raises(TypeError, match="positions_are_validated"):
        ForwardBatch(
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0]),
            positions_are_validated=1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "query_start_loc",
    ((1, 3), (0, 2), (0, 2, 2, 3), (0, 3.0)),
)
def test_forward_batch_rejects_invalid_query_boundaries(
    query_start_loc: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="query_start_loc"):
        ForwardBatch(
            input_ids=torch.tensor([1, 2, 3]),
            query_start_loc=query_start_loc,
        )


def test_forward_batch_exposes_packed_query_lengths() -> None:
    batch = ForwardBatch(
        input_ids=torch.tensor([1, 2, 3, 4, 5]),
        positions=torch.tensor([0, 1, 2, 0, 1]),
        query_start_loc=(0, 3, 5),
    )

    assert batch.batch_size == 2
    assert batch.query_lengths == (3, 2)


def test_forward_batch_requires_positions_for_multiple_requests() -> None:
    with pytest.raises(ValueError, match="positions are required"):
        ForwardBatch(
            input_ids=torch.tensor([1, 2, 3]),
            query_start_loc=(0, 1, 3),
        )


def test_model_projects_only_requested_query_rows() -> None:
    runner = create_runner()
    runner.load(tiny_spec())
    session = runner.open_session()
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6])
    positions = torch.tensor([0, 1, 2, 0, 1, 2])
    full = session.forward(
        ForwardBatch(
            input_ids=input_ids,
            positions=positions,
            query_start_loc=(0, 3, 6),
        )
    ).logits

    selected = session.forward(
        ForwardBatch(
            input_ids=input_ids,
            positions=positions,
            query_start_loc=(0, 3, 6),
            logit_query_indices=(2, 4),
        )
    ).logits
    skipped = session.forward(
        ForwardBatch(
            input_ids=input_ids,
            positions=positions,
            query_start_loc=(0, 3, 6),
            logit_query_indices=(),
        )
    ).logits

    assert selected.shape == (2, 32)
    torch.testing.assert_close(selected[0], full[2])
    torch.testing.assert_close(selected[1], full[4])
    assert skipped.shape == (0, 32)


def test_model_reuses_states_when_every_query_row_is_requested() -> None:
    hidden_states = torch.randn(3, 8)
    batch = ForwardBatch(
        input_ids=torch.tensor([1, 2, 3]),
        logit_query_indices=(0, 1, 2),
    )

    assert select_query_states(hidden_states, batch) is hidden_states


def test_forward_batch_rejects_invalid_logit_query_indices() -> None:
    with pytest.raises(ValueError, match="valid token rows"):
        ForwardBatch(
            input_ids=torch.tensor([1, 2]),
            logit_query_indices=(2,),
        )


def test_attention_model_exposes_its_kv_cache_shape_without_runner_dispatch() -> None:
    runner = create_runner()
    runner.load(
        ModelSpec(
            architecture="tiny-attention-causal-lm",
            model_args={"vocab_size": 32, "hidden_size": 8, "num_heads": 2},
        )
    )

    assert runner.open_session().kv_cache_spec == ModelKVCacheSpec(
        layers=(
            AttentionLayerSpec(
                layer_id="attention",
                num_query_heads=2,
                num_kv_heads=2,
                head_size=4,
            ),
        )
    )


def test_state_dict_loader_round_trip(tmp_path: Path) -> None:
    spec = tiny_spec()
    source_model = create_catalog().models.get(spec.architecture)(spec).eval()
    batch = ForwardBatch(input_ids=torch.tensor([1, 2]))
    source_output = source_model(batch)

    checkpoint = tmp_path / "tiny.pt"
    torch.save(source_model.state_dict(), checkpoint)

    restored = create_runner()
    restored.load(tiny_spec(loader="state-dict", weights=checkpoint))
    restored_output = restored.open_session().forward(batch)

    torch.testing.assert_close(restored_output.logits, source_output.logits)


class ScaleModel(nn.Module):
    def forward(self, batch: ForwardBatch) -> ModelOutput:
        return ModelOutput(logits=batch.input_ids.float().unsqueeze(-1) * 2)


class DirectLoader:
    def load(self, spec: ModelSpec, factory: ModelFactory) -> nn.Module:
        return factory(spec).eval()


def test_components_can_be_added_without_core_changes() -> None:
    catalog = create_catalog()
    catalog.models.register("scale", lambda spec: ScaleModel())
    catalog.loaders.register("direct", DirectLoader())

    runner = create_runner(catalog)
    runner.load(ModelSpec(architecture="scale", loader="direct"))
    output = runner.open_session().forward(ForwardBatch(input_ids=torch.tensor([2, 4])))

    torch.testing.assert_close(output.logits, torch.tensor([[4.0], [8.0]]))


def test_failed_reload_keeps_current_model() -> None:
    runner = create_runner()
    runner.load(tiny_spec())

    with pytest.raises(RuntimeError):
        runner.load(tiny_spec(loader="state-dict", weights=None))

    assert runner.generation == 1
    output = runner.open_session().forward(ForwardBatch(input_ids=torch.tensor([1])))
    assert output.logits.shape == (1, 32)


def test_open_session_keeps_one_model_generation_across_reload() -> None:
    runner = create_runner()
    runner.load(tiny_spec(model_args={"vocab_size": 32, "hidden_size": 8}))
    old_session = runner.open_session()

    runner.load(tiny_spec(model_args={"vocab_size": 16, "hidden_size": 8}))

    batch = ForwardBatch(input_ids=torch.tensor([1]))
    assert old_session.generation == 1
    assert old_session.forward(batch).logits.shape[-1] == 32
    assert runner.open_session().forward(batch).logits.shape[-1] == 16
