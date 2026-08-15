from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from light_vllm import ForwardBatch, ModelSpec, create_runner
from light_vllm.modeling.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
from light_vllm.runtime.execution.dense_attention import (
    DenseAttentionMetadata,
    TorchDenseAttention,
)
from light_vllm.runtime.execution.paged_attention import (
    PagedAttentionMetadata,
    TorchPagedAttentionBackend,
)
from light_vllm.runtime.execution.paged_cache import PagedKVCache, PagedKVCacheConfig


def qwen2_args(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "model_type": "qwen2",
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "rope_theta": 1_000_000.0,
        "tie_word_embeddings": True,
    }
    values.update(changes)
    return values


def _loaded_qwen2():
    torch.manual_seed(7)
    runner = create_runner()
    runner.load(ModelSpec(architecture="qwen2", model_args=qwen2_args()))
    return runner.open_session()


def _dense_forward(session, input_ids: torch.Tensor, *, past=None):
    assert session.kv_cache_spec is not None
    positions = torch.arange(input_ids.shape[1]).expand(input_ids.shape[0], -1)
    if past is not None:
        positions = positions + past.num_tokens
    attention = TorchDenseAttention(
        session.kv_cache_spec,
        DenseAttentionMetadata(
            positions=positions,
            query_lengths=(input_ids.shape[1],) * input_ids.shape[0],
        ),
        past,
    )
    output = session.forward(
        ForwardBatch(
            input_ids=input_ids,
            positions=positions,
            attention=attention,
        )
    )
    return output, attention


def test_qwen2_exposes_model_limits_and_kv_shape() -> None:
    session = _loaded_qwen2()

    assert session.max_model_tokens == 32
    assert session.kv_cache_spec is not None
    assert tuple(layer.layer_id for layer in session.kv_cache_spec.layers) == (
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
    )
    assert all(layer.num_query_heads == 4 for layer in session.kv_cache_spec.layers)
    assert all(layer.num_kv_heads == 2 for layer in session.kv_cache_spec.layers)


def test_qwen2_contiguous_kv_matches_full_sequence() -> None:
    session = _loaded_qwen2()
    input_ids = torch.tensor([[1, 5, 9, 13]])
    full, _ = _dense_forward(session, input_ids)

    prefill, prefill_attention = _dense_forward(session, input_ids[:, :3])
    decode, _ = _dense_forward(
        session,
        input_ids[:, 3:],
        past=prefill_attention.cache_updates,
    )

    torch.testing.assert_close(prefill.logits, full.logits[:, :3], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(decode.logits, full.logits[:, 3:], atol=1e-6, rtol=1e-5)


def test_qwen2_paged_attention_matches_full_sequence() -> None:
    session = _loaded_qwen2()
    assert session.kv_cache_spec is not None
    input_ids = torch.tensor([[2, 4, 6, 8]])
    full, _ = _dense_forward(session, input_ids)
    cache = PagedKVCache(
        session.kv_cache_spec,
        PagedKVCacheConfig(
            num_blocks=4,
            block_size=2,
            dtype=torch.float32,
            device=torch.device("cpu"),
        ),
    )
    attention = TorchPagedAttentionBackend().create(
        cache,
        PagedAttentionMetadata(
            block_tables=((3, 1),),
            num_computed_tokens=(0,),
            query_lengths=(4,),
        ),
    )

    paged = session.forward(ForwardBatch(input_ids=input_ids, attention=attention))

    assert attention.layer_ids == frozenset(
        {"model.layers.0.self_attn", "model.layers.1.self_attn"}
    )
    torch.testing.assert_close(paged.logits, full.logits, atol=1e-6, rtol=1e-5)


def test_qwen2_config_rejects_features_not_implemented_yet() -> None:
    with pytest.raises(ValueError, match="sliding-window"):
        Qwen2Config.from_mapping(qwen2_args(use_sliding_window=True))

    with pytest.raises(ValueError, match="rope_scaling"):
        Qwen2Config.from_mapping(qwen2_args(rope_scaling={"type": "linear"}))


def _write_snapshot(
    path: Path,
    config: dict[str, object],
    model: Qwen2ForCausalLM,
    *,
    sharded: bool,
) -> None:
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    if config["tie_word_embeddings"]:
        # HF 保存共享权重时通常只保留其中一个名字。
        state.pop("lm_head.weight")
    if not sharded:
        save_file(state, path / "model.safetensors")
        return

    names = sorted(state)
    middle = len(names) // 2
    shard_names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    save_file({name: state[name] for name in names[:middle]}, path / shard_names[0])
    save_file({name: state[name] for name in names[middle:]}, path / shard_names[1])
    weight_map = {
        name: shard_names[0] if index < middle else shard_names[1]
        for index, name in enumerate(names)
    }
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )


@pytest.mark.parametrize(("sharded", "tied"), ((False, True), (True, False)))
def test_qwen2_loads_hf_compatible_safetensors_snapshot(
    tmp_path: Path,
    sharded: bool,
    tied: bool,
) -> None:
    config_values = qwen2_args(tie_word_embeddings=tied)
    torch.manual_seed(11)
    source = Qwen2ForCausalLM(Qwen2Config.from_mapping(config_values)).eval()
    input_ids = torch.tensor([[3, 1, 4]])
    expected, _ = _dense_forward(source, input_ids)
    _write_snapshot(tmp_path, config_values, source, sharded=sharded)

    runner = create_runner()
    runner.load(
        ModelSpec(
            architecture="qwen2",
            loader="safetensors",
            weights=tmp_path,
        )
    )
    actual, _ = _dense_forward(runner.open_session(), input_ids)

    torch.testing.assert_close(actual.logits, expected.logits)


def test_qwen2_logits_match_transformers_reference() -> None:
    transformers = pytest.importorskip("transformers")
    config_values = qwen2_args(tie_word_embeddings=False)
    hf_config = transformers.Qwen2Config(**config_values)
    hf_config._attn_implementation = "eager"
    torch.manual_seed(19)
    reference = transformers.Qwen2ForCausalLM(hf_config).eval()
    model = Qwen2ForCausalLM(Qwen2Config.from_mapping(config_values)).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    input_ids = torch.tensor([[1, 2, 3, 5, 8]])

    with torch.inference_mode():
        expected = reference(input_ids=input_ids, use_cache=False).logits
        actual = _dense_forward(model, input_ids)[0].logits

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
