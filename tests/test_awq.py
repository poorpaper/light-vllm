from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from light_vllm import ForwardBatch, ModelSpec, create_runner
from light_vllm.entrypoints.http import create_serving_app
from light_vllm.modeling.loaders.torch import ModelLoadError
from light_vllm.modeling.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
from light_vllm.modeling.quantization.awq import (
    AWQColumnParallelLinear,
    AWQConfig,
    AWQLinearMethod,
    AWQRowParallelLinear,
    dequantize_awq,
    pack_awq,
    quantize_awq_weight,
    unpack_awq,
)
from light_vllm.modeling.quantization.awq_export import (
    atomic_output_directory,
    copy_qwen2_checkpoint_assets,
    export_awq_checkpoint,
)
from light_vllm.modeling.quantization.awq_ptq import AWQPTQConfig
from light_vllm.modeling.quantization.awq_schemes import (
    TorchAWQScheme,
    select_awq_scheme,
)
from light_vllm.modeling.tensor_parallel import (
    TensorParallelContext,
    checkpoint_shards,
)
from light_vllm.runtime.execution.dense_attention import (
    DenseAttentionMetadata,
    TorchDenseAttention,
)
from light_vllm.runtime.execution.layout import linear_query_layout


def _config() -> dict[str, object]:
    return {
        "model_type": "qwen2",
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "tie_word_embeddings": False,
    }


def _write_dense_checkpoint(
    path: Path,
    *,
    quantization_config: dict[str, object] | None = None,
) -> None:
    config = _config()
    if quantization_config is not None:
        config["quantization_config"] = quantization_config
    model = Qwen2ForCausalLM(Qwen2Config.from_mapping(config)).half().eval()
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(model.state_dict(), path / "model.safetensors")


class _LocalShardCollectives:
    def all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class _TokenDecoder:
    def push(self, token_id: int) -> str:
        return f"<{token_id}>"

    def finish(self) -> str:
        return ""


class _TextProcessor:
    eos_token_id = 63
    vocab_size = 64

    def encode_prompt(self, prompt: str) -> tuple[int, ...]:
        return (1, 2) if prompt else ()

    def encode_chat(self, messages) -> tuple[int, ...]:
        return (1, 2) if messages else ()

    def new_decoder(self) -> _TokenDecoder:
        return _TokenDecoder()


def _quantized_module_names() -> tuple[str, ...]:
    return tuple(
        f"model.layers.0.{name}"
        for name in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    )

    def all_gather_last_dim(
        self,
        tensor: torch.Tensor,
        partition_sizes: tuple[int, ...],
    ) -> torch.Tensor:
        return tensor


def _forward(model: Qwen2ForCausalLM, token_ids: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(token_ids.numel())
    attention = TorchDenseAttention(
        model.kv_cache_spec,
        DenseAttentionMetadata(
            positions=positions,
            query_layouts=(linear_query_layout(token_ids.numel()),),
        ),
    )
    return model(
        ForwardBatch(
            input_ids=token_ids,
            positions=positions,
            attention=attention,
        )
    ).logits


def test_auto_awq_scheme_uses_torch_on_cpu() -> None:
    assert isinstance(select_awq_scheme("auto", device="cpu"), TorchAWQScheme)


def test_explicit_cuda_awq_rejects_cpu() -> None:
    with pytest.raises(ValueError, match="requires a CUDA device"):
        select_awq_scheme("cuda", device="cpu")


def test_awq_pack_round_trip_uses_autoawq_column_order() -> None:
    values = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [15, 14, 13, 12, 11, 10, 9, 8],
        ],
        dtype=torch.int32,
    )

    packed = pack_awq(values)

    # bit slot 按 0,2,4,6,1,3,5,7 写入，固定常量可以抓住“自己打包、
    # 自己解包”同时写错而测试仍通过的问题。
    assert packed[0, 0].item() == 0x75316420
    torch.testing.assert_close(unpack_awq(packed), values)


def test_awq_quantized_weight_matches_its_dequantized_reference() -> None:
    torch.manual_seed(3)
    weight = torch.randn(16, 24, dtype=torch.float16)

    quantized = quantize_awq_weight(weight, group_size=8)
    restored = dequantize_awq(
        quantized.qweight,
        quantized.qzeros,
        quantized.scales,
        group_size=8,
    )

    torch.testing.assert_close(restored.t(), quantized.dequantized)
    assert quantized.qweight.shape == (24, 2)
    assert quantized.qzeros.shape == (3, 2)
    assert quantized.scales.shape == (3, 16)


def test_awq_parallel_layers_declare_packed_checkpoint_shards() -> None:
    context = TensorParallelContext()
    config = AWQConfig(group_size=8)
    column = AWQColumnParallelLinear(16, 32, context, config, bias=True)
    row = AWQRowParallelLinear(32, 16, context, config, bias=False)

    class Layers(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.column = column
            self.row = row

    shards = checkpoint_shards(Layers())

    assert shards["column.qweight"].full_size == 4
    assert shards["column.scales"].full_size == 32
    assert shards["row.qweight"].full_size == 32
    assert shards["row.qzeros"].full_size == 4


def test_awq_tp2_column_and_row_shards_match_full_dequantized_math() -> None:
    torch.manual_seed(6)
    config = AWQConfig(group_size=8)
    column_weight = quantize_awq_weight(torch.randn(16, 16), group_size=8)
    row_weight = quantize_awq_weight(torch.randn(16, 16), group_size=8)
    column_inputs = torch.randn(5, 16)
    row_inputs = torch.randn(5, 16)
    column_outputs = []
    row_outputs = []
    for rank in range(2):
        context = TensorParallelContext(rank, 2, _LocalShardCollectives())
        column = AWQColumnParallelLinear(16, 16, context, config, bias=False)
        column.qweight.copy_(column_weight.qweight[:, rank : rank + 1])
        column.qzeros.copy_(column_weight.qzeros[:, rank : rank + 1])
        column.scales.copy_(column_weight.scales[:, rank * 8 : (rank + 1) * 8])
        column_outputs.append(column(column_inputs))

        row = AWQRowParallelLinear(16, 16, context, config, bias=False)
        row.qweight.copy_(row_weight.qweight[rank * 8 : (rank + 1) * 8])
        row.qzeros.copy_(row_weight.qzeros[rank : rank + 1])
        row.scales.copy_(row_weight.scales[rank : rank + 1])
        row_outputs.append(row(row_inputs[:, rank * 8 : (rank + 1) * 8]))

    torch.testing.assert_close(
        torch.cat(column_outputs, dim=-1),
        column_inputs @ column_weight.dequantized.t(),
    )
    torch.testing.assert_close(
        sum(row_outputs),
        row_inputs @ row_weight.dequantized.t(),
    )


def test_safetensors_loader_detects_awq_and_matches_pseudo_quantized_model(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    config_values = _config()
    source = Qwen2ForCausalLM(Qwen2Config.from_mapping(config_values)).half().eval()
    source_state = {name: value.detach().clone() for name, value in source.state_dict().items()}
    checkpoint: dict[str, torch.Tensor] = {}
    reference_state: dict[str, torch.Tensor] = {}
    quantized_suffixes = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "down_proj.weight",
    )
    for name, value in source_state.items():
        if name.endswith(quantized_suffixes):
            packed = quantize_awq_weight(value, group_size=8)
            prefix = name.removesuffix(".weight")
            checkpoint[f"{prefix}.qweight"] = packed.qweight
            checkpoint[f"{prefix}.qzeros"] = packed.qzeros
            checkpoint[f"{prefix}.scales"] = packed.scales
            reference_state[name] = packed.dequantized
        else:
            checkpoint[name] = value
            reference_state[name] = value

    config_values["quantization_config"] = {
        "quant_method": "awq",
        "bits": 4,
        "group_size": 8,
        "zero_point": True,
        "version": "GEMM",
    }
    (tmp_path / "config.json").write_text(json.dumps(config_values), encoding="utf-8")
    save_file(checkpoint, tmp_path / "model.safetensors")

    reference = Qwen2ForCausalLM(Qwen2Config.from_mapping(config_values)).half().eval()
    reference.load_state_dict(reference_state)
    runner = create_runner()
    runner.load(
        ModelSpec(
            architecture="qwen2",
            loader="safetensors",
            weights=tmp_path,
            dtype=torch.float16,
        )
    )
    loaded = runner.open_session()
    tokens = torch.tensor([1, 2, 3, 5])

    with torch.inference_mode():
        expected = _forward(reference, tokens)
        actual = loaded.forward(
            ForwardBatch(
                input_ids=tokens,
                positions=torch.arange(tokens.numel()),
                attention=TorchDenseAttention(
                    loaded.kv_cache_spec,
                    DenseAttentionMetadata(
                        positions=torch.arange(tokens.numel()),
                        query_layouts=(linear_query_layout(tokens.numel()),),
                    ),
                ),
            )
        ).logits

    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    awq_model = loaded._model  # type: ignore[attr-defined]
    assert isinstance(awq_model.model.layers[0].self_attn.q_proj, AWQColumnParallelLinear)
    assert isinstance(awq_model.model.layers[0].mlp.down_proj, AWQRowParallelLinear)


@pytest.mark.parametrize(
    ("checkpoint_quantization", "requested", "message"),
    (
        (None, "awq", "requires checkpoint quantization metadata"),
        ({"quant_method": "awq"}, "none", "cannot load a quantized checkpoint"),
        ({"quant_method": "unknown"}, "auto", "cannot configure quantization 'unknown'"),
    ),
)
def test_safetensors_loader_rejects_incompatible_quantization_selection(
    tmp_path: Path,
    checkpoint_quantization: dict[str, object] | None,
    requested: str,
    message: str,
) -> None:
    _write_dense_checkpoint(tmp_path, quantization_config=checkpoint_quantization)

    with pytest.raises(ModelLoadError, match=message):
        create_runner().load(
            ModelSpec(
                architecture="qwen2",
                loader="safetensors",
                weights=tmp_path,
                dtype=torch.float16,
                quantization=requested,
            )
        )


def test_awq_method_reuses_merged_storage_without_duplicate_qweight() -> None:
    method = AWQLinearMethod(AWQConfig(group_size=8), TorchAWQScheme())
    context = TensorParallelContext()
    first = method.create_column(16, 16, context, prefix="first", bias=False)
    second = method.create_column(16, 8, context, prefix="second", bias=False)
    assert isinstance(first, AWQColumnParallelLinear)
    assert isinstance(second, AWQColumnParallelLinear)

    operation = method.prepare_merged((first, second))

    assert first.qweight.untyped_storage().data_ptr() == second.qweight.untyped_storage().data_ptr()
    assert operation is not None


def test_awq_export_writes_sharded_checkpoint_loadable_by_runtime(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = Qwen2ForCausalLM(Qwen2Config.from_mapping(_config())).half().eval()

    export_awq_checkpoint(
        model,
        tmp_path,
        _quantized_module_names(),
        group_size=8,
        ptq_config=AWQPTQConfig(group_size=8),
        calibration={"samples": 2, "token_ids_sha256": "fixture"},
        max_shard_bytes=1024,
    )
    runner = create_runner()
    runner.load(
        ModelSpec(
            architecture="qwen2",
            loader="safetensors",
            weights=tmp_path,
            dtype=torch.float16,
        )
    )

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text("utf-8"))
    assert len(set(index["weight_map"].values())) > 1
    assert "model.layers.0.self_attn.q_proj.qweight" in index["weight_map"]
    report = json.loads((tmp_path / "awq_ptq_report.json").read_text("utf-8"))
    assert report["ptq_config"]["group_size"] == 8
    assert report["calibration"] == {"samples": 2, "token_ids_sha256": "fixture"}
    assert runner.open_session().tensor_parallel_size == 1


def test_awq_export_loads_the_expected_tp2_packed_slices(tmp_path: Path) -> None:
    torch.manual_seed(13)
    config = _config()
    # 每 Rank 至少持有 8 个输出值，才能对应一个完整 AutoAWQ int32 pack。
    config["num_key_value_heads"] = 4
    model = Qwen2ForCausalLM(Qwen2Config.from_mapping(config)).half().eval()
    expected_q = quantize_awq_weight(
        model.model.layers[0].self_attn.q_proj.weight.detach(),
        group_size=8,
    )
    export_awq_checkpoint(
        model,
        tmp_path,
        _quantized_module_names(),
        group_size=8,
    )

    loaded = []
    for rank in range(2):
        runner = create_runner()
        runner.load(
            ModelSpec(
                architecture="qwen2",
                loader="safetensors",
                weights=tmp_path,
                dtype=torch.float16,
                tensor_parallel=TensorParallelContext(rank, 2, _LocalShardCollectives()),
            )
        )
        loaded.append(runner.open_session()._model)  # type: ignore[attr-defined]

    for rank, rank_model in enumerate(loaded):
        q_proj = rank_model.model.layers[0].self_attn.q_proj
        torch.testing.assert_close(q_proj.qweight, expected_q.qweight[:, rank : rank + 1])
        torch.testing.assert_close(
            q_proj.scales,
            expected_q.scales[:, rank * 8 : (rank + 1) * 8],
        )


def test_awq_checkpoint_serves_through_the_openai_completion_adapter(tmp_path: Path) -> None:
    torch.manual_seed(17)
    model = Qwen2ForCausalLM(Qwen2Config.from_mapping(_config())).half().eval()
    export_awq_checkpoint(
        model,
        tmp_path,
        _quantized_module_names(),
        group_size=8,
    )
    app = create_serving_app(
        ModelSpec(
            architecture="qwen2",
            loader="safetensors",
            weights=tmp_path,
            dtype=torch.float16,
        ),
        text_processor=_TextProcessor(),
        served_model_name="tiny-qwen-awq",
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "tiny-qwen-awq",
                "prompt": "hello",
                "max_tokens": 1,
                "temperature": 0,
            },
        )

    assert response.status_code == 200
    assert response.json()["model"] == "tiny-qwen-awq"
    assert response.json()["usage"]["prompt_tokens"] == 2


def test_awq_output_directory_is_published_only_after_success(tmp_path: Path) -> None:
    output = tmp_path / "published"

    with atomic_output_directory(output) as staging:
        (staging / "complete.txt").write_text("ready", encoding="utf-8")
        assert not output.exists()

    assert (output / "complete.txt").read_text(encoding="utf-8") == "ready"


def test_awq_output_directory_cleans_failed_staging(tmp_path: Path) -> None:
    output = tmp_path / "failed"

    with pytest.raises(RuntimeError, match="stop"), atomic_output_directory(output) as staging:
        (staging / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("stop")

    assert not output.exists()
    assert not tuple(tmp_path.glob(".failed.staging-*"))


def test_awq_export_copies_tokenizer_assets_without_reserializing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    tokenizer_bytes = b'{"pre_tokenizer":{"regex":"exact-source-bytes"}}\n'
    (source / "tokenizer.json").write_bytes(tokenizer_bytes)
    (source / "tokenizer_config.json").write_text('{"chat_template":"exact"}', "utf-8")

    copied = copy_qwen2_checkpoint_assets(source, output)

    assert copied == ("tokenizer.json", "tokenizer_config.json")
    assert (output / "tokenizer.json").read_bytes() == tokenizer_bytes
