from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from light_vllm.entrypoints.quantize_awq import _parser as awq_cli_parser
from light_vllm.entrypoints.quantize_awq import main as quantize_awq_main
from light_vllm.modeling.quantization.awq_ptq import (
    AWQPTQConfig,
    apply_awq_clip,
    apply_awq_scale,
    capture_qwen2_first_layer_inputs,
    pseudo_quantize_weight,
    quantize_qwen2_layers,
    search_awq_clip,
    search_awq_scale,
)
from light_vllm.modeling.quantization.calibration import (
    build_calibration_input_ids,
    read_calibration_texts,
)


def test_awq_scale_application_preserves_dense_math() -> None:
    torch.manual_seed(3)
    norm = nn.LayerNorm(16, elementwise_affine=True, bias=False)
    linear = nn.Linear(16, 24, bias=True)
    inputs = torch.randn(7, 16)
    scales = torch.linspace(0.5, 1.5, 16)
    expected = linear(norm(inputs))

    apply_awq_scale(norm, (linear,), scales)
    actual = linear(norm(inputs))

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_awq_scale_search_is_no_worse_than_unscaled_pseudo_quantization() -> None:
    torch.manual_seed(5)
    linear = nn.Linear(16, 24, bias=False)
    inputs = torch.randn(32, 16) * torch.linspace(0.05, 7, 16)
    reference = linear(inputs)
    baseline = F.linear(inputs, pseudo_quantize_weight(linear.weight, 8))
    baseline_error = (baseline - reference).float().pow(2).mean().item()

    result = search_awq_scale(inputs, (linear,), group_size=8, num_steps=20)

    assert 0 <= result.ratio < 1
    assert result.error <= baseline_error + 1e-7


def test_awq_scale_search_uses_the_composed_module_loss_and_restores_weights() -> None:
    class GatedMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = nn.Linear(16, 24, bias=False)
            self.up = nn.Linear(16, 24, bias=False)
            self.max_batch = 0

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            self.max_batch = max(self.max_batch, values.shape[0])
            return F.silu(self.gate(values)) * self.up(values)

    torch.manual_seed(6)
    module = GatedMLP()
    inputs = torch.randn(32, 16) * torch.linspace(0.1, 5, 16)
    original_gate = module.gate.weight.detach().clone()
    original_up = module.up.weight.detach().clone()
    reference = module(inputs)
    baseline = F.silu(F.linear(inputs, pseudo_quantize_weight(original_gate, 8))) * F.linear(
        inputs, pseudo_quantize_weight(original_up, 8)
    )
    baseline_error = (baseline - reference).float().pow(2).mean().item()
    module.max_batch = 0

    result = search_awq_scale(
        inputs,
        (module.gate, module.up),
        group_size=8,
        num_steps=10,
        inspect_module=module,
        max_parallel_samples=3,
    )

    assert result.error <= baseline_error + 1e-7
    assert module.max_batch == 3
    torch.testing.assert_close(module.gate.weight, original_gate)
    torch.testing.assert_close(module.up.weight, original_up)


def test_awq_clip_search_keeps_a_valid_candidate_for_every_group() -> None:
    torch.manual_seed(7)
    weight = torch.randn(24, 16)
    weight[0, 0] = 12
    inputs = torch.randn(64, 16)
    result = search_awq_clip(weight, inputs, group_size=8, num_steps=10, max_tokens=32)
    clipped = weight.clone()
    apply_awq_clip(clipped, result.max_values, 8)

    assert result.max_values.shape == (24, 2, 1)
    original_max = weight.reshape(24, 2, 8).abs().amax(dim=-1, keepdim=True)
    assert torch.all(result.max_values <= original_max)
    assert torch.isfinite(torch.tensor(result.mean_error))


def test_calibration_reader_accepts_text_and_sharegpt_jsonl(tmp_path) -> None:
    text_path = tmp_path / "code.txt"
    text_path.write_text("def add(a, b): return a + b\n\nprint(add(1, 2))\n", encoding="utf-8")
    jsonl_path = tmp_path / "chat.jsonl"
    jsonl_path.write_text(
        '{"conversations":[{"from":"human","value":"写一个栈"},'
        '{"from":"gpt","value":"可以用 list"}]}\n',
        encoding="utf-8",
    )

    texts = read_calibration_texts(text_path) + read_calibration_texts(jsonl_path)

    assert texts[0].startswith("def add")
    assert "<human>" in texts[-1]


def test_calibration_reader_streams_a_sharegpt_json_array(tmp_path) -> None:
    json_path = tmp_path / "sharegpt.json"
    # 用很小的 chunk 强制对象和字符串跨 chunk，覆盖真实大文件的增量路径。
    json_path.write_text(
        '[{"conversations":[{"from":"human","value":"first message"}]},{"text":"second message"}]',
        encoding="utf-8",
    )

    from light_vllm.modeling.quantization.calibration import _json_array_records

    with json_path.open("r", encoding="utf-8") as stream:
        records = tuple(_json_array_records(stream, chunk_size=7))
    texts = read_calibration_texts(json_path)

    assert len(records) == 2
    assert texts == ("<human>\nfirst message", "second message")


@pytest.mark.parametrize(
    "payload",
    (
        '[{"text":"ok"},]',
        '[{"text":"ok"}] trailing',
        '[{"text":"missing close"}',
    ),
)
def test_calibration_json_array_rejects_ambiguous_partial_files(tmp_path, payload: str) -> None:
    path = tmp_path / "broken.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError):
        read_calibration_texts(path)


def test_calibration_token_builder_uses_full_blocks_without_padding() -> None:
    class Tokenizer:
        eos_token_id = 99

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            assert not add_special_tokens
            return [int(value) for value in text.split()]

    result = build_calibration_input_ids(
        Tokenizer(),
        ("1 2 3", "4 5 6 7 8"),
        max_samples=2,
        sequence_length=4,
    )

    assert result.tolist() == [[1, 2, 3, 99], [4, 5, 6, 7]]


def test_qwen2_ptq_orchestrator_captures_and_searches_each_projection() -> None:
    transformers = __import__("transformers")
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = transformers.Qwen2ForCausalLM(config).half().eval()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    hidden_states, kwargs = capture_qwen2_first_layer_inputs(
        model,
        input_ids,
        device="cpu",
        max_parallel_samples=1,
    )
    names, reports = quantize_qwen2_layers(
        model,
        hidden_states,
        kwargs,
        AWQPTQConfig(
            group_size=8,
            num_scale_steps=2,
            num_clip_steps=2,
            max_clip_tokens=4,
            max_parallel_calibration_samples=1,
        ),
        device="cpu",
    )

    assert len(names) == 7
    assert reports[0].layer_index == 0
    assert set(reports[0].scale_ratios) == {
        "attention_input",
        "mlp_input",
        "mlp_output",
    }


def test_qwen2_ptq_skips_only_the_requested_projection() -> None:
    transformers = __import__("transformers")
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = transformers.Qwen2ForCausalLM(config).half().eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    hidden_states, kwargs = capture_qwen2_first_layer_inputs(model, input_ids, device="cpu")

    names, _ = quantize_qwen2_layers(
        model,
        hidden_states,
        kwargs,
        AWQPTQConfig(
            group_size=8,
            num_scale_steps=2,
            num_clip_steps=2,
            max_clip_tokens=4,
            modules_to_not_convert=("self_attn.q_proj",),
        ),
        device="cpu",
    )

    assert len(names) == 6
    assert "model.layers.0.self_attn.q_proj" not in names


@pytest.mark.parametrize(
    "kwargs",
    (
        {"max_parallel_calibration_samples": 0},
        {"modules_to_not_convert": ("",)},
    ),
)
def test_awq_ptq_config_rejects_invalid_calibration_controls(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AWQPTQConfig(**kwargs)


def test_awq_cli_exposes_bounded_calibration_controls() -> None:
    args = awq_cli_parser().parse_args(
        [
            "--model",
            "dense-model",
            "--output",
            "awq-model",
            "--calibration-data",
            "samples.jsonl",
            "--max-parallel-calibration-samples",
            "3",
            "--max-clip-shrink",
            "0.75",
        ]
    )

    assert args.max_parallel_calibration_samples == 3
    assert args.max_clip_shrink == 0.75


def test_awq_cli_rejects_an_existing_output_before_loading_the_model(tmp_path) -> None:
    with pytest.raises(FileExistsError, match="already exists"):
        quantize_awq_main(
            [
                "--model",
                str(tmp_path / "unused-model"),
                "--output",
                str(tmp_path),
                "--calibration-data",
                str(tmp_path / "unused-data"),
                "--device",
                "cpu",
            ]
        )
