import pytest

from light_vllm.runtime.execution import (
    DraftTree,
    ExecutionRequest,
    ModelStepRequest,
    QueryLayout,
)
from light_vllm.runtime.execution.layout import (
    append_draft_layout,
    linear_model_step_request,
    linear_query_layout,
    query_visibility,
    semantic_positions,
)


def test_draft_tree_requires_parents_to_precede_children() -> None:
    with pytest.raises(ValueError, match="precede"):
        DraftTree((4, 5), (-1, 1))


def test_draft_tree_rejects_duplicate_sibling_tokens() -> None:
    with pytest.raises(ValueError, match="siblings"):
        DraftTree((4, 4), (-1, -1))


def test_query_layout_derives_tree_positions_and_ancestor_visibility() -> None:
    layout = QueryLayout((-1, 0, 0, 2))

    assert semantic_positions(layout, prefix_length=5) == (5, 6, 6, 7)
    assert query_visibility(layout) == (
        (True, False, False, False),
        (True, True, False, False),
        (True, False, True, False),
        (True, False, True, True),
    )


def test_append_draft_layout_connects_every_root_to_the_formal_input_tail() -> None:
    draft = DraftTree((7, 9, 8), (-1, 0, 0))

    layout = append_draft_layout(2, draft)

    assert layout.parent_indices == (-1, 0, 1, 2, 2)


def test_linear_model_step_keeps_unused_slots_as_reservation_only() -> None:
    request = ExecutionRequest(
        request_id="request",
        input_token_ids=(4, 5),
        context_token_ids=(1, 2, 4, 5),
        num_computed_tokens=2,
        num_lookahead_tokens=3,
        max_output_tokens=4,
        block_ids=(3, 1),
    )

    model_request = linear_model_step_request(request)

    assert model_request.query_token_ids == (4, 5)
    assert model_request.query_layout == linear_query_layout(2)
    assert model_request.num_reserved_query_tokens == 5


def test_model_step_rejects_layout_or_reservation_smaller_than_actual_query() -> None:
    with pytest.raises(ValueError, match="reserved"):
        ModelStepRequest(
            request_id="request",
            query_token_ids=(1, 2),
            num_computed_tokens=0,
            num_reserved_query_tokens=1,
            query_layout=linear_query_layout(2),
            block_ids=None,
        )

    with pytest.raises(ValueError, match="one parent"):
        ModelStepRequest(
            request_id="request",
            query_token_ids=(1, 2),
            num_computed_tokens=0,
            num_reserved_query_tokens=2,
            query_layout=linear_query_layout(1),
            block_ids=None,
        )
