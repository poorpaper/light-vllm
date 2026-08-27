from __future__ import annotations

import json
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.nn import functional as F

from light_vllm.entrypoints.http import create_serving_app
from light_vllm.modeling.attention.interfaces import AttentionLayerSpec, ModelKVCacheSpec
from light_vllm.modeling.loaders.safetensors import SafetensorsModelLoader
from light_vllm.modeling.models.interfaces import ModelSpec
from light_vllm.modeling.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
from light_vllm.modeling.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelContext,
    VocabParallelEmbedding,
    VocabParallelLinear,
    checkpoint_shards,
    partition_dimension,
)
from light_vllm.runtime.execution.distributed import (
    SynchronizedPagedKVCachePlanner,
    TensorParallelModelExecutor,
    TorchDistributedGroup,
    _SocketCommandChannel,
)
from light_vllm.runtime.execution.interfaces import (
    ExecutionBatch,
    ExecutionCapabilities,
    ExecutionError,
    ExecutionOutput,
    ExecutionRequest,
    RequestOutput,
)
from light_vllm.runtime.execution.nccl import CurrentStreamNCCL
from light_vllm.runtime.execution.paged_cache import PagedKVCacheConfig


class _LocalShardCollectives:
    """单进程测试只观察本 Rank 结果，最后由测试显式合并。"""

    def all_reduce_sum(self, tensor: Tensor) -> Tensor:
        return tensor

    def all_gather_last_dim(
        self,
        tensor: Tensor,
        partition_sizes: tuple[int, ...],
    ) -> Tensor:
        return tensor


class _ForbiddenCollectives:
    """TP=1 快路径不应触发任何集合通信。"""

    def all_reduce_sum(self, tensor: Tensor) -> Tensor:
        raise AssertionError("TP=1 must not all-reduce embedding output")

    def all_gather_last_dim(
        self,
        tensor: Tensor,
        partition_sizes: tuple[int, ...],
    ) -> Tensor:
        raise AssertionError("TP=1 must not all-gather model output")


class _FakeLeaderGroup:
    rank = 0
    world_size = 2

    def __init__(self) -> None:
        self.commands: list[str] = []

    def broadcast_object(self, value: object | None, *, src: int = 0) -> object:
        assert src == 0
        assert value is not None
        self.commands.append(type(value).__name__)
        return value

    def all_gather_object(self, value: object) -> tuple[object, ...]:
        return value, value

    def first_rank(self, selected: bool) -> int | None:
        assert not selected
        return None


class _FakeCapacityGroup:
    def minimum(self, value: int) -> int:
        assert value == 11
        return 7


class _FakeLease:
    def release(self) -> None:
        return None


class _FakeExecutor:
    def __init__(self) -> None:
        self.ready = False
        self.capabilities = ExecutionCapabilities(32, 64, tensor_parallel_size=2)
        self.requests: set[str] = set()

    def initialize(self) -> None:
        self.ready = True

    def add_request(self, request_id: str, *, capacity: int) -> None:
        assert capacity == 8
        self.requests.add(request_id)

    def free_request(self, request_id: str) -> bool:
        if request_id not in self.requests:
            return False
        self.requests.remove(request_id)
        return True

    def acquire(self, request_ids: tuple[str, ...]) -> _FakeLease:
        assert set(request_ids) <= self.requests
        return _FakeLease()

    def execute(self, batch: ExecutionBatch) -> ExecutionOutput:
        request = batch.requests[0]
        return ExecutionOutput(
            requests=(
                RequestOutput(
                    request_id=request.request_id,
                    num_input_tokens_computed=len(request.input_token_ids),
                    output_token_ids=(9,),
                ),
            ),
            num_model_tokens_computed=len(request.input_token_ids),
        )


def _context(rank: int, world_size: int = 2) -> TensorParallelContext:
    return TensorParallelContext(rank, world_size, _LocalShardCollectives())


def _qwen_config(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "model_type": "qwen2",
        "vocab_size": 10,
        "hidden_size": 8,
        "intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 16,
        "tie_word_embeddings": False,
    }
    values.update(changes)
    return values


def test_partition_dimension_keeps_every_element_once() -> None:
    first = partition_dimension(5, 0, 2)
    second = partition_dimension(5, 1, 2)

    assert first.sizes == (3, 2)
    assert (first.start, first.end) == (0, 3)
    assert (second.start, second.end) == (3, 5)


def test_parallel_linear_embedding_and_lm_head_match_full_math() -> None:
    torch.manual_seed(3)
    inputs = torch.randn(4, 6)
    column_weight = torch.randn(8, 6)
    row_weight = torch.randn(5, 8)
    embedding_weight = torch.randn(10, 6)
    token_ids = torch.tensor([0, 4, 5, 9])

    column_outputs = []
    row_outputs = []
    embedding_outputs = []
    lm_head_outputs = []
    column_input = torch.randn(4, 8)
    lm_hidden = torch.randn(4, 6)
    for rank in range(2):
        context = _context(rank)
        output_partition = context.partition(8)
        vocab_partition = context.partition(10)

        column = ColumnParallelLinear(6, 8, context, bias=False)
        column.weight.data.copy_(column_weight[output_partition.start : output_partition.end])
        column_outputs.append(column(inputs))

        row = RowParallelLinear(8, 5, context, bias=False)
        row.weight.data.copy_(row_weight[:, output_partition.start : output_partition.end])
        row_outputs.append(row(column_input[:, output_partition.start : output_partition.end]))

        embedding = VocabParallelEmbedding(10, 6, context)
        embedding.weight.data.copy_(embedding_weight[vocab_partition.start : vocab_partition.end])
        embedding_outputs.append(embedding(token_ids))

        lm_head = VocabParallelLinear(6, 10, context)
        lm_head.weight.data.copy_(embedding_weight[vocab_partition.start : vocab_partition.end])
        lm_head_outputs.append(lm_head(lm_hidden))

    torch.testing.assert_close(torch.cat(column_outputs, dim=-1), F.linear(inputs, column_weight))
    torch.testing.assert_close(sum(row_outputs), F.linear(column_input, row_weight))
    torch.testing.assert_close(sum(embedding_outputs), F.embedding(token_ids, embedding_weight))
    torch.testing.assert_close(
        torch.cat(lm_head_outputs, dim=-1),
        F.linear(lm_hidden, embedding_weight),
    )


def test_single_rank_embedding_uses_the_local_embedding_path() -> None:
    context = TensorParallelContext(0, 1, _ForbiddenCollectives())
    embedding = VocabParallelEmbedding(4, 2, context)
    embedding.weight.data.copy_(torch.arange(8, dtype=torch.float32).reshape(4, 2))

    output = embedding(torch.tensor([0, 3]))

    torch.testing.assert_close(output, embedding.weight[[0, 3]])


def test_qwen_parallel_layers_declare_checkpoint_slices() -> None:
    model = Qwen2ForCausalLM(
        Qwen2Config.from_mapping(_qwen_config()),
        parallel=_context(1),
    )
    shards = checkpoint_shards(model)

    assert model.tensor_parallel_size == 2
    assert model.kv_cache_spec.layers[0].num_query_heads == 2
    assert model.kv_cache_spec.layers[0].num_kv_heads == 1
    assert shards["model.layers.0.self_attn.q_proj.weight"].dimension == 0
    assert shards["model.layers.0.self_attn.o_proj.weight"].dimension == 1
    assert shards["model.layers.0.mlp.down_proj.weight"].dimension == 1
    assert shards["model.embed_tokens.weight"].start == 5


def test_qwen_replicates_kv_heads_when_there_are_fewer_heads_than_ranks() -> None:
    models = tuple(
        Qwen2ForCausalLM(
            Qwen2Config.from_mapping(_qwen_config(num_key_value_heads=1)),
            parallel=_context(rank),
        )
        for rank in range(2)
    )
    shards = tuple(checkpoint_shards(model) for model in models)

    assert all(model.kv_cache_spec.layers[0].num_kv_heads == 1 for model in models)
    assert all(value["model.layers.0.self_attn.k_proj.weight"].start == 0 for value in shards)
    assert all(value["model.layers.0.self_attn.k_proj.weight"].length == 2 for value in shards)


def test_safetensors_loader_copies_only_each_rank_checkpoint_slice(tmp_path: Path) -> None:
    config_values = _qwen_config()
    torch.manual_seed(5)
    source = Qwen2ForCausalLM(Qwen2Config.from_mapping(config_values)).eval()
    source_state = {name: value.detach().clone() for name, value in source.state_dict().items()}
    (tmp_path / "config.json").write_text(json.dumps(config_values), encoding="utf-8")
    save_file(source_state, tmp_path / "model.safetensors")

    loaded: list[nn.Module] = []
    for rank in range(2):
        spec = ModelSpec(
            architecture="qwen2",
            loader="safetensors",
            weights=tmp_path,
            tensor_parallel=_context(rank),
        )
        loaded.append(SafetensorsModelLoader().load(spec, Qwen2ForCausalLM.from_spec))

    q_weight = source_state["model.layers.0.self_attn.q_proj.weight"]
    o_weight = source_state["model.layers.0.self_attn.o_proj.weight"]
    embedding_weight = source_state["model.embed_tokens.weight"]
    for rank, model in enumerate(loaded):
        assert isinstance(model, Qwen2ForCausalLM)
        torch.testing.assert_close(
            model.model.layers[0].self_attn.q_proj.weight,
            q_weight[rank * 4 : (rank + 1) * 4],
        )
        torch.testing.assert_close(
            model.model.layers[0].self_attn.o_proj.weight,
            o_weight[:, rank * 4 : (rank + 1) * 4],
        )
        torch.testing.assert_close(
            model.model.embed_tokens.weight,
            embedding_weight[rank * 5 : (rank + 1) * 5],
        )


def test_distributed_kv_planner_uses_the_smallest_rank_capacity() -> None:
    local = PagedKVCacheConfig(num_blocks=11, block_size=4)
    planner = SynchronizedPagedKVCachePlanner(local, _FakeCapacityGroup())  # type: ignore[arg-type]
    model_spec = ModelKVCacheSpec(
        layers=(AttentionLayerSpec("layer", 2, 1, 4),),
    )

    config = planner.plan(model_spec)

    assert config.num_blocks == 7
    assert planner.num_blocks == 7
    assert planner.block_size == 4
    assert planner.device == torch.device("cpu")


def test_current_stream_nccl_uses_one_communicator_and_rank_major_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object, int]] = []
    communicator = object()
    stream_handle = 12345

    class FakeStream:
        cuda_stream = stream_handle

    class FakeLibrary:
        def all_reduce_sum(
            self,
            tensor: Tensor,
            current_communicator: object,
            stream: int,
        ) -> None:
            calls.append(("reduce", current_communicator, stream))
            tensor.mul_(2)

        def all_gather(
            self,
            tensor: Tensor,
            output: Tensor,
            current_communicator: object,
            stream: int,
        ) -> None:
            calls.append(("gather", current_communicator, stream))
            output[0].copy_(tensor)
            output[1].copy_(tensor + 10)

        def destroy(self, current_communicator: object) -> None:
            calls.append(("destroy", current_communicator, 0))

    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: FakeStream())
    collectives = CurrentStreamNCCL(FakeLibrary(), communicator, 2)

    reduced = torch.tensor([[1.0, 2.0]])
    assert collectives.all_reduce_sum(reduced) is reduced
    torch.testing.assert_close(reduced, torch.tensor([[2.0, 4.0]]))
    gathered = collectives.all_gather(torch.tensor([[3.0, 4.0]]))
    torch.testing.assert_close(
        gathered,
        torch.tensor([[[3.0, 4.0]], [[13.0, 14.0]]]),
    )
    collectives.close()
    collectives.close()
    assert calls == [
        ("reduce", communicator, stream_handle),
        ("gather", communicator, stream_handle),
        ("destroy", communicator, 0),
    ]


def test_distributed_group_close_releases_all_resources_after_channel_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    destroyed: list[object] = []
    control_group = object()
    world_group = object()

    class FailingChannel:
        def close(self) -> None:
            events.append("command")
            raise RuntimeError("command close failed")

    class FakeNCCL:
        def close(self) -> None:
            events.append("nccl")

    def destroy_process_group(group: object = world_group) -> None:
        destroyed.append(group)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "destroy_process_group", destroy_process_group)
    group = TorchDistributedGroup(
        rank=0,
        world_size=2,
        local_rank=0,
        device=torch.device("cpu"),
        control_group=control_group,  # type: ignore[arg-type]
        tensor_group=world_group,  # type: ignore[arg-type]
        current_stream_nccl=FakeNCCL(),  # type: ignore[arg-type]
        owns_world_group=True,
    )
    group._command_channel = FailingChannel()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="command close failed"):
        group.close()

    assert events == ["command", "nccl"]
    assert destroyed == [control_group, world_group]


def test_local_socket_command_channel_preserves_rank_order() -> None:
    driver_socket, worker_socket = socket.socketpair()
    driver = _SocketCommandChannel(rank=0, peers={1: driver_socket})
    worker = _SocketCommandChannel(rank=1, peers={0: worker_socket})
    command = {"request_ids": tuple(f"request-{index}" for index in range(16))}

    def run_worker() -> object:
        received = worker.broadcast(None)
        worker.complete({"rank": 1}, failed=False, require_consensus=False)
        return received

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run_worker)
            assert driver.broadcast(command) == command
            results = driver.complete(
                {"rank": 0},
                failed=False,
                require_consensus=False,
            )
            assert future.result() == command
        assert results == ({"rank": 0}, {"rank": 1})
    finally:
        worker.close()
        driver.close()


def test_local_socket_command_channel_reports_a_closed_peer() -> None:
    driver_socket, worker_socket = socket.socketpair()
    worker = _SocketCommandChannel(rank=1, peers={0: worker_socket})
    driver_socket.close()

    try:
        with pytest.raises(ExecutionError, match="peer exited"):
            worker.broadcast(None)
    finally:
        worker.close()


def test_tensor_parallel_executor_delivers_lifecycle_at_safe_boundaries() -> None:
    local = _FakeExecutor()
    group = _FakeLeaderGroup()
    executor = TensorParallelModelExecutor(local, group)  # type: ignore[arg-type]
    batch = ExecutionBatch(
        requests=(
            ExecutionRequest(
                request_id="request-1",
                input_token_ids=(1, 2),
                context_token_ids=None,
                num_computed_tokens=0,
                num_lookahead_tokens=0,
                max_output_tokens=1,
                block_ids=None,
            ),
        ),
    )

    executor.initialize()
    executor.add_request("request-1", capacity=8)
    assert group.commands == ["_Initialize"]

    output = executor.execute(batch)

    assert output.requests[0].output_token_ids == (9,)
    assert group.commands == ["_Initialize", "_Execute"]
    assert executor.free_request("request-1")
    executor.shutdown()
    assert group.commands == [
        "_Initialize",
        "_Execute",
        "_UpdateRequests",
        "_Shutdown",
    ]
    executor.shutdown()
    with pytest.raises(ExecutionError, match="closed"):
        executor.add_request("request-2", capacity=8)


def test_tensor_parallel_executor_defers_free_until_a_prepared_batch_finishes() -> None:
    local = _FakeExecutor()
    group = _FakeLeaderGroup()
    executor = TensorParallelModelExecutor(local, group)  # type: ignore[arg-type]
    batch = ExecutionBatch(
        requests=(
            ExecutionRequest(
                request_id="request-1",
                input_token_ids=(1, 2),
                context_token_ids=None,
                num_computed_tokens=0,
                num_lookahead_tokens=0,
                max_output_tokens=1,
                block_ids=None,
            ),
        ),
    )
    executor.initialize()
    executor.add_request("request-1", capacity=8)
    lease = executor.acquire(("request-1",))

    assert executor.free_request("request-1")
    assert "request-1" in local.requests
    assert executor.execute(batch).requests[0].output_token_ids == (9,)

    lease.release()
    assert "request-1" not in local.requests
    executor.shutdown()


def test_tensor_parallel_lifecycle_updates_do_not_wait_for_a_model_step() -> None:
    local = _FakeExecutor()
    group = _FakeLeaderGroup()
    executor = TensorParallelModelExecutor(local, group)  # type: ignore[arg-type]
    batch = ExecutionBatch(
        requests=(
            ExecutionRequest(
                request_id="request-1",
                input_token_ids=(1, 2),
                context_token_ids=None,
                num_computed_tokens=0,
                num_lookahead_tokens=0,
                max_output_tokens=1,
                block_ids=None,
            ),
        ),
    )
    entered_step = Event()
    release_step = Event()
    free_started = Event()
    original_execute = local.execute

    def blocking_execute(value: ExecutionBatch) -> ExecutionOutput:
        entered_step.set()
        assert release_step.wait(timeout=1)
        return original_execute(value)

    def free_request() -> bool:
        free_started.set()
        return executor.free_request("request-1")

    local.execute = blocking_execute  # type: ignore[method-assign]
    executor.initialize()
    executor.add_request("request-1", capacity=8)
    lease = executor.acquire(("request-1",))

    with ThreadPoolExecutor(max_workers=3) as pool:
        execute_future = pool.submit(executor.execute, batch)
        assert entered_step.wait(timeout=1)
        free_future = pool.submit(free_request)
        assert free_started.wait(timeout=1)
        try:
            assert free_future.result(timeout=1)
            add_future = pool.submit(executor.add_request, "request-2", capacity=8)
            add_future.result(timeout=1)
            assert local.requests == {"request-1", "request-2"}
        finally:
            release_step.set()
        assert execute_future.result(timeout=1).requests[0].output_token_ids == (9,)

    lease.release()
    assert local.requests == {"request-2"}
    assert executor.free_request("request-2")
    executor.shutdown()


def test_tensor_parallel_serving_rejects_unbounded_request_kv() -> None:
    group = _FakeLeaderGroup()
    parallel = TensorParallelContext(0, 2, group)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="requires paged KV"):
        create_serving_app(
            ModelSpec(
                architecture="tiny-attention-causal-lm",
                model_args={"vocab_size": 16, "hidden_size": 4, "num_heads": 1},
                tensor_parallel=parallel,
            ),
            runtime="engine",
            kv_reservation="unbounded",
            tensor_parallel_group=group,  # type: ignore[arg-type]
        )


def test_distributed_group_rejects_multi_node_torchrun_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")

    with pytest.raises(ExecutionError, match="one node"):
        TorchDistributedGroup.initialize(backend="gloo")
