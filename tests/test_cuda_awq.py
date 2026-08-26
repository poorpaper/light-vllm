from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from light_vllm.modeling.quantization.awq import quantize_awq_weight
from light_vllm.modeling.quantization.cuda_awq import (
    cuda_awq_available,
    cuda_awq_linear,
)

pytestmark = pytest.mark.skipif(
    not cuda_awq_available(),
    reason="CUDA AWQ validation requires a CUDA device",
)


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 256])
@pytest.mark.parametrize("inner", [128, 384])
def test_cuda_awq_matches_dequantized_torch(num_tokens: int, inner: int) -> None:
    torch.manual_seed(3)
    weight = torch.randn(128, inner, dtype=torch.float16, device="cuda")
    packed = quantize_awq_weight(weight, group_size=32)
    inputs = torch.randn(num_tokens, inner, dtype=torch.float16, device="cuda")
    bias = torch.randn(128, dtype=torch.float16, device="cuda")
    expected = F.linear(inputs, packed.dequantized, bias)

    actual = cuda_awq_linear(
        inputs,
        packed.qweight,
        packed.qzeros,
        packed.scales,
        32,
        bias,
    )

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=2e-2)


def test_cuda_awq_supports_tp_shard_narrower_than_group_size() -> None:
    torch.manual_seed(7)
    weight = torch.randn(64, 896, dtype=torch.float16, device="cuda")
    packed = quantize_awq_weight(weight, group_size=128)
    inputs = torch.randn(7, 896, dtype=torch.float16, device="cuda")
    expected = F.linear(inputs, packed.dequantized)

    actual = cuda_awq_linear(
        inputs,
        packed.qweight,
        packed.qzeros,
        packed.scales,
        128,
        None,
    )

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=2e-2)
