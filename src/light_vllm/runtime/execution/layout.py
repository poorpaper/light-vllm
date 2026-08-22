"""从紧凑父链推导模型步骤所需的位置与可见性。"""

from __future__ import annotations

from light_vllm.runtime.execution.interfaces import (
    DraftTree,
    ExecutionRequest,
    ModelStepRequest,
    QueryLayout,
)


def linear_query_layout(num_queries: int) -> QueryLayout:
    """构造与标准 causal attention 等价的线性父链。"""

    if type(num_queries) is not int or num_queries <= 0:
        raise ValueError("num_queries must be a positive integer")
    return QueryLayout(tuple(-1 if index == 0 else index - 1 for index in range(num_queries)))


def append_draft_layout(num_input_queries: int, draft: DraftTree) -> QueryLayout:
    """把草稿根接到正式输入末尾，形成一次完整 model step。"""

    if type(num_input_queries) is not int or num_input_queries <= 0:
        raise ValueError("num_input_queries must be a positive integer")
    parents = list(linear_query_layout(num_input_queries).parent_indices)
    for parent in draft.parent_indices:
        parents.append(num_input_queries - 1 if parent == -1 else num_input_queries + parent)
    return QueryLayout(tuple(parents))


def semantic_positions(layout: QueryLayout, *, prefix_length: int) -> tuple[int, ...]:
    """按父链深度计算 RoPE 使用的请求内绝对位置。"""

    if type(prefix_length) is not int or prefix_length < 0:
        raise ValueError("prefix_length must be a non-negative integer")
    positions: list[int] = []
    for parent in layout.parent_indices:
        positions.append(prefix_length if parent == -1 else positions[parent] + 1)
    return tuple(positions)


def query_visibility(layout: QueryLayout) -> tuple[tuple[bool, ...], ...]:
    """返回每个 query 对自身和祖先 query 的可见关系。"""

    width = len(layout)
    rows: list[tuple[bool, ...]] = []
    for query_index, parent in enumerate(layout.parent_indices):
        visible = [False] * width
        visible[query_index] = True
        while parent != -1:
            visible[parent] = True
            parent = layout.parent_indices[parent]
        rows.append(tuple(visible))
    return tuple(rows)


def linear_model_step_request(request: ExecutionRequest) -> ModelStepRequest:
    """把 Engine 的正式输入转换成不含策略信息的线性模型步骤。"""

    return ModelStepRequest(
        request_id=request.request_id,
        query_token_ids=request.input_token_ids,
        num_computed_tokens=request.num_computed_tokens,
        num_reserved_query_tokens=(len(request.input_token_ids) + request.num_lookahead_tokens),
        query_layout=linear_query_layout(len(request.input_token_ids)),
        block_ids=request.block_ids,
        num_readonly_prefix_blocks=request.num_readonly_prefix_blocks,
        logit_query_indices=(len(request.input_token_ids) - 1,)
        if request.max_output_tokens
        else (),
    )
