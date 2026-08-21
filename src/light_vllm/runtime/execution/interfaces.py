from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from torch import Tensor

    from light_vllm.modeling.models.interfaces import ModelSession


class ExecutionError(RuntimeError):
    """执行一次模型计算失败时抛出。"""


class ExecutionNotReadyError(ExecutionError):
    """模型执行器还没有准备好时抛出。"""


@dataclass(frozen=True, slots=True)
class AcceptanceResult:
    """候选验收后可见的输出，以及真正命中的草稿路径。"""

    output_token_ids: tuple[int, ...]
    accepted_draft_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        output_token_ids = tuple(self.output_token_ids)
        accepted_draft_indices = tuple(self.accepted_draft_indices)
        if not output_token_ids:
            raise ValueError("acceptance result must contain at least one output token")
        if any(type(token_id) is not int or token_id < 0 for token_id in output_token_ids):
            raise ValueError("output_token_ids must contain non-negative integers")
        if any(type(index) is not int or index < 0 for index in accepted_draft_indices):
            raise ValueError("accepted_draft_indices must contain non-negative integers")
        if len(set(accepted_draft_indices)) != len(accepted_draft_indices):
            raise ValueError("accepted draft indices must be unique")
        if len(output_token_ids) != len(accepted_draft_indices) + 1:
            raise ValueError("acceptance result must leave one uncached output token")
        object.__setattr__(self, "output_token_ids", output_token_ids)
        object.__setattr__(self, "accepted_draft_indices", accepted_draft_indices)

    @property
    def num_cached_output_tokens(self) -> int:
        return len(self.accepted_draft_indices)


@dataclass(frozen=True, slots=True)
class DraftTree:
    """按父节点优先顺序展平的草稿树。"""

    token_ids: tuple[int, ...]
    parent_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        token_ids = tuple(self.token_ids)
        parent_indices = tuple(self.parent_indices)
        if len(token_ids) != len(parent_indices):
            raise ValueError("draft tokens and parents must have the same length")
        if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
            raise ValueError("draft tokens must be non-negative integers")
        siblings: set[tuple[int, int]] = set()
        for index, (token_id, parent) in enumerate(zip(token_ids, parent_indices, strict=True)):
            if type(parent) is not int or not -1 <= parent < index:
                raise ValueError("each draft parent must be -1 or precede its child")
            sibling = (parent, token_id)
            if sibling in siblings:
                raise ValueError("draft siblings must use distinct token IDs")
            siblings.add(sibling)
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "parent_indices", parent_indices)

    def __len__(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class QueryLayout:
    """一次模型步骤内每个 query 对更早 query 的父依赖。"""

    parent_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        parent_indices = tuple(self.parent_indices)
        if not parent_indices:
            raise ValueError("query layout must not be empty")
        for index, parent in enumerate(parent_indices):
            if type(parent) is not int or not -1 <= parent < index:
                raise ValueError("each query parent must be -1 or precede its child")
        object.__setattr__(self, "parent_indices", parent_indices)

    def __len__(self) -> int:
        return len(self.parent_indices)


class DraftProposer(Protocol):
    """根据完整已知 token 历史提出一棵有界草稿树。"""

    def propose(self, token_ids: tuple[int, ...], *, max_nodes: int) -> DraftTree: ...


class AcceptanceSampler(Protocol):
    """对照目标模型结果，决定哪些候选可以确认。"""

    def accept(
        self,
        draft: DraftTree,
        target_token_ids: tuple[int, ...],
    ) -> AcceptanceResult: ...


@dataclass(frozen=True, slots=True)
class SpeculativeDecodeObservation:
    """一次成功树验证产生的候选、验证产出和 KV 搬运事实。"""

    num_proposed_nodes: int
    num_accepted_nodes: int
    # 包含命中的草稿节点和 target model 最终补出的一个 token。
    num_verified_tokens: int
    num_draft_roots: int
    num_branching_parents: int
    max_draft_depth: int
    num_compacted_tokens: int

    def __post_init__(self) -> None:
        values = (
            self.num_proposed_nodes,
            self.num_accepted_nodes,
            self.num_verified_tokens,
            self.num_draft_roots,
            self.num_branching_parents,
            self.max_draft_depth,
            self.num_compacted_tokens,
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("speculative decode counters must be non-negative integers")
        if self.num_accepted_nodes > self.num_proposed_nodes:
            raise ValueError("accepted nodes must not exceed proposed nodes")
        if self.num_verified_tokens != self.num_accepted_nodes + 1:
            raise ValueError("verified tokens must contain accepted nodes plus one target token")
        if self.num_draft_roots > self.num_proposed_nodes:
            raise ValueError("draft roots must not exceed proposed nodes")
        if self.num_branching_parents > self.num_proposed_nodes:
            raise ValueError("branching parents must not exceed proposed nodes")
        if self.max_draft_depth > self.num_proposed_nodes:
            raise ValueError("draft depth must not exceed proposed nodes")
        if self.num_compacted_tokens > self.num_accepted_nodes:
            raise ValueError("compacted tokens must not exceed accepted nodes")


class SpeculationObserver(Protocol):
    """接收投机解码事实的轻量旁路端口。"""

    def speculation_completed(
        self,
        observation: SpeculativeDecodeObservation,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ExecutionCapabilities:
    """执行器初始化后能够支持的模型长度和 KV cache 容量。"""

    max_model_tokens: int | None
    max_kv_cache_tokens: int | None
    # 物理 KV 重新创建时递增；prefix cache 不能跨 epoch 复用旧页。
    kv_cache_epoch: int | None = None


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """一个请求本轮交给模型计算的输入。

    ``input_token_ids`` 是已经确定的 token。``num_lookahead_tokens`` 只表示
    为投机解码提前留出的 KV 位置；这些位置的 token 还没有确定，由执行器
    产生候选 token 并返回最终通过验证的结果。
    """

    request_id: str
    input_token_ids: tuple[int, ...]
    # 只有草稿 proposer 需要从 prompt 开始的完整只读 token 快照；普通执行不携带它。
    context_token_ids: tuple[int, ...] | None
    num_computed_tokens: int
    num_lookahead_tokens: int
    max_output_tokens: int
    # 分页 KV cache 需要 block table；连续 KV cache 不需要，使用 None。
    block_ids: tuple[int, ...] | None
    num_readonly_prefix_blocks: int = 0

    def __post_init__(self) -> None:
        input_token_ids = tuple(self.input_token_ids)
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not input_token_ids:
            raise ValueError("input_token_ids must not be empty")
        if any(type(token_id) is not int or token_id < 0 for token_id in input_token_ids):
            raise ValueError("input_token_ids must contain non-negative integers")
        if type(self.num_computed_tokens) is not int or self.num_computed_tokens < 0:
            raise ValueError("num_computed_tokens must be a non-negative integer")
        if type(self.num_lookahead_tokens) is not int or self.num_lookahead_tokens < 0:
            raise ValueError("num_lookahead_tokens must be a non-negative integer")
        if type(self.max_output_tokens) is not int or self.max_output_tokens < 0:
            raise ValueError("max_output_tokens must be a non-negative integer")
        if type(self.num_readonly_prefix_blocks) is not int or self.num_readonly_prefix_blocks < 0:
            raise ValueError("num_readonly_prefix_blocks must be a non-negative integer")
        object.__setattr__(self, "input_token_ids", input_token_ids)
        if self.context_token_ids is not None:
            context_token_ids = tuple(self.context_token_ids)
            if not context_token_ids or any(
                type(token_id) is not int or token_id < 0 for token_id in context_token_ids
            ):
                raise ValueError("context_token_ids must contain non-negative integers")
            input_end = self.num_computed_tokens + len(input_token_ids)
            if tuple(context_token_ids[self.num_computed_tokens : input_end]) != input_token_ids:
                raise ValueError("input_token_ids must be the scheduled slice of context_token_ids")
            object.__setattr__(self, "context_token_ids", context_token_ids)
        if self.block_ids is None and self.num_readonly_prefix_blocks:
            raise ValueError("readonly prefix blocks require a block table")
        if self.block_ids is not None:
            object.__setattr__(self, "block_ids", tuple(self.block_ids))
            if self.num_readonly_prefix_blocks > len(self.block_ids):
                raise ValueError("readonly prefix blocks must fit within the block table")


@dataclass(frozen=True, slots=True)
class ExecutionBatch:
    """一次模型调用需要执行的所有请求。"""

    requests: tuple[ExecutionRequest, ...]

    def __post_init__(self) -> None:
        requests = tuple(self.requests)
        request_ids = tuple(request.request_id for request in requests)
        if not requests:
            raise ValueError("execution batch must not be empty")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("execution batch request IDs must be unique")
        object.__setattr__(self, "requests", requests)

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self.requests)


@dataclass(frozen=True, slots=True)
class ModelStepRequest:
    """Decode Handler 已经确定的一次具体模型计算。"""

    request_id: str
    query_token_ids: tuple[int, ...]
    num_computed_tokens: int
    # 包含正式输入和 speculative slots；未使用的槽位不进入模型 forward。
    num_reserved_query_tokens: int
    query_layout: QueryLayout
    block_ids: tuple[int, ...] | None
    num_readonly_prefix_blocks: int = 0
    # None requests logits for every query. Decode handlers set the exact rows
    # needed for sampling so prefill-only work can skip the vocabulary head.
    logit_query_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        query_token_ids = tuple(self.query_token_ids)
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not query_token_ids:
            raise ValueError("query_token_ids must not be empty")
        if any(type(token_id) is not int or token_id < 0 for token_id in query_token_ids):
            raise ValueError("query_token_ids must contain non-negative integers")
        if type(self.num_computed_tokens) is not int or self.num_computed_tokens < 0:
            raise ValueError("num_computed_tokens must be a non-negative integer")
        if type(self.num_reserved_query_tokens) is not int or self.num_reserved_query_tokens < len(
            query_token_ids
        ):
            raise ValueError("reserved query tokens must cover every actual query")
        if len(self.query_layout) != len(query_token_ids):
            raise ValueError("query layout must contain one parent per query token")
        if type(self.num_readonly_prefix_blocks) is not int or self.num_readonly_prefix_blocks < 0:
            raise ValueError("num_readonly_prefix_blocks must be a non-negative integer")
        logit_query_indices = self.logit_query_indices
        if logit_query_indices is not None:
            logit_query_indices = tuple(logit_query_indices)
            if any(
                type(index) is not int or not 0 <= index < len(query_token_ids)
                for index in logit_query_indices
            ):
                raise ValueError("logit query indices must select valid query positions")
        object.__setattr__(self, "query_token_ids", query_token_ids)
        object.__setattr__(self, "logit_query_indices", logit_query_indices)
        if self.block_ids is None and self.num_readonly_prefix_blocks:
            raise ValueError("readonly prefix blocks require a block table")
        if self.block_ids is not None:
            block_ids = tuple(self.block_ids)
            if self.num_readonly_prefix_blocks > len(block_ids):
                raise ValueError("readonly prefix blocks must fit within the block table")
            object.__setattr__(self, "block_ids", block_ids)


@dataclass(frozen=True, slots=True)
class ModelStepBatch:
    """一次目标模型 forward 包含的具体 query 请求。"""

    requests: tuple[ModelStepRequest, ...]

    def __post_init__(self) -> None:
        requests = tuple(self.requests)
        request_ids = tuple(request.request_id for request in requests)
        if not requests:
            raise ValueError("model step batch must not be empty")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("model step request IDs must be unique")
        object.__setattr__(self, "requests", requests)

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(request.request_id for request in self.requests)


@dataclass(frozen=True, slots=True)
class ModelStepOutput:
    """一次模型步骤的连续 logits，以及按请求切分它的边界。"""

    logits: Tensor
    logits_start_loc: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.logits.ndim != 2:
            raise ValueError("model step logits must have shape [rows, vocabulary]")
        starts = tuple(self.logits_start_loc)
        if (
            len(starts) < 2
            or starts[0] != 0
            or starts[-1] != self.logits.shape[0]
            or any(type(index) is not int for index in starts)
            or any(left > right for left, right in zip(starts, starts[1:], strict=False))
        ):
            raise ValueError("logits_start_loc must cover every logits row in request order")
        object.__setattr__(self, "logits_start_loc", starts)

    @property
    def num_requests(self) -> int:
        return len(self.logits_start_loc) - 1

    def request_logits(self, request_index: int) -> Tensor:
        if type(request_index) is not int or not 0 <= request_index < self.num_requests:
            raise IndexError("model step request index is out of range")
        start = self.logits_start_loc[request_index]
        end = self.logits_start_loc[request_index + 1]
        return self.logits[start:end]


@dataclass(frozen=True, slots=True)
class RequestOutput:
    """一个请求本轮算完了多少输入，以及最终确认了哪些新 token。

    ``output_token_ids`` 的前 ``num_cached_output_tokens`` 个 token 已经写入
    KV cache；其余 token 虽然已经确认，但下一轮仍要作为输入再计算一次。
    普通解码通常返回一个尚未写入缓存的 token；投机解码可以一次返回多个。
    """

    request_id: str
    num_input_tokens_computed: int
    output_token_ids: tuple[int, ...] = ()
    num_cached_output_tokens: int = 0

    def __post_init__(self) -> None:
        output_token_ids = tuple(self.output_token_ids)
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if type(self.num_input_tokens_computed) is not int or self.num_input_tokens_computed < 0:
            raise ValueError("num_input_tokens_computed must be a non-negative integer")
        if any(type(token_id) is not int or token_id < 0 for token_id in output_token_ids):
            raise ValueError("output_token_ids must contain non-negative integers")
        if type(
            self.num_cached_output_tokens
        ) is not int or not 0 <= self.num_cached_output_tokens <= len(output_token_ids):
            raise ValueError("num_cached_output_tokens must be within output_token_ids")
        object.__setattr__(self, "output_token_ids", output_token_ids)


@dataclass(frozen=True, slots=True)
class ExecutionOutput:
    """一次模型执行的请求结果、实际模型计算量与设备完成耗时。

    ``num_model_tokens_computed`` 统计真正送入模型 forward 的 query token；
    投机 lookahead 只预留位置，proposer 未实际产出的部分不能计入该值。
    """

    requests: tuple[RequestOutput, ...]
    num_model_tokens_computed: int
    step_elapsed_seconds: float | None = None

    def __post_init__(self) -> None:
        requests = tuple(self.requests)
        request_ids = tuple(request.request_id for request in requests)
        if not requests:
            raise ValueError("execution output must not be empty")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("execution output request IDs must be unique")
        if type(self.num_model_tokens_computed) is not int or self.num_model_tokens_computed <= 0:
            raise ValueError("num_model_tokens_computed must be a positive integer")
        if self.step_elapsed_seconds is not None and (
            not isfinite(self.step_elapsed_seconds) or self.step_elapsed_seconds < 0
        ):
            raise ValueError("step_elapsed_seconds must be finite and non-negative")
        object.__setattr__(self, "requests", requests)


class ExecutionTimer(Protocol):
    """测量一次完整设备步骤；具体后端负责定义完成边界。"""

    def measure(
        self,
        operation: Callable[[], ExecutionOutput],
    ) -> tuple[ExecutionOutput, float]: ...


class ExecutionLease(Protocol):
    """保证模型计算结束前，请求使用的缓存不会被释放或复用。"""

    def release(self) -> None: ...


class TokenExecutionSession(Protocol):
    """参考实现选定模型后，用来逐个计算新 token 的接口。"""

    def next_token(self, token_ids: tuple[int, ...]) -> int: ...


class TokenExecutor(Protocol):
    """为一次参考生成选定模型，并创建逐 token 执行对象。"""

    @property
    def ready(self) -> bool: ...

    def open_session(self) -> TokenExecutionSession: ...


class ModelExecutor(Protocol):
    """Engine 调用模型的统一入口。

    Executor 隐藏本进程、多进程或分布式等执行方式，并可以协调多个 Worker。
    """

    @property
    def ready(self) -> bool: ...

    @property
    def capabilities(self) -> ExecutionCapabilities: ...

    def initialize(self) -> None: ...

    def add_request(self, request_id: str, *, capacity: int) -> None: ...

    def free_request(self, request_id: str) -> bool: ...

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease: ...

    def execute(self, batch: ExecutionBatch) -> ExecutionOutput: ...


class ModelStepHandler(Protocol):
    """执行一次已经确定 token 的模型计算，并管理对应的物理 KV cache。"""

    @property
    def max_kv_cache_tokens(self) -> int | None: ...

    def add_request(self, request_id: str, *, capacity: int) -> None: ...

    def free_request(self, request_id: str) -> None: ...

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease: ...

    def forward(
        self,
        model: ModelSession,
        batch: ModelStepBatch,
    ) -> ModelStepOutput:
        """返回连续 logits，并保留按请求切分所需的边界。"""

        ...

    def compact(
        self,
        request: ModelStepRequest,
        retained_query_indices: tuple[int, ...],
    ) -> int:
        """按正式 token 顺序保留 query K/V，并返回实际搬运的 token 数。"""

        ...


class DecodeHandler(Protocol):
    """组织无输出输入步骤、普通解码或投机解码，并产出确认 token。"""

    def execute(
        self,
        model: ModelSession,
        batch: ExecutionBatch,
        step: ModelStepHandler,
    ) -> ExecutionOutput: ...


class ModelWorker(Protocol):
    """一个设备或分布式 rank 内的模型执行单元。"""

    @property
    def ready(self) -> bool: ...

    @property
    def capabilities(self) -> ExecutionCapabilities: ...

    def initialize(self) -> None: ...

    def add_request(self, request_id: str, *, capacity: int) -> None: ...

    def free_request(self, request_id: str) -> bool: ...

    def acquire(self, request_ids: tuple[str, ...]) -> ExecutionLease: ...

    def execute(self, batch: ExecutionBatch) -> ExecutionOutput: ...
