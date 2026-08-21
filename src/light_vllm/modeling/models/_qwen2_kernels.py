"""Optional fused inference kernels used by the native Qwen2 model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU and Windows installs do not require Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _fused_add_rms_norm_kernel(
        hidden_ptr,
        residual_ptr,
        weight_ptr,
        row_stride,
        width,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < width
        hidden_offsets = row * row_stride + offsets
        values = tl.load(hidden_ptr + hidden_offsets, mask=mask, other=0.0).to(tl.float32)
        values += tl.load(
            residual_ptr + hidden_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        variance = tl.sum(values * values, axis=0) / width
        normalized = values * tl.rsqrt(variance + eps)
        weights = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(hidden_ptr + hidden_offsets, normalized * weights, mask=mask)
        tl.store(residual_ptr + hidden_offsets, values, mask=mask)

    @triton.jit
    def _silu_and_mul_kernel(
        gate_ptr,
        up_ptr,
        row_stride,
        width,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < width
        tensor_offsets = row * row_stride + offsets
        gate = tl.load(gate_ptr + tensor_offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + tensor_offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            gate_ptr + tensor_offsets,
            gate * tl.sigmoid(gate) * up,
            mask=mask,
        )

    @triton.jit
    def _rotary_embedding_kernel(
        query_ptr,
        key_ptr,
        cosine_ptr,
        sine_ptr,
        query_token_stride,
        query_head_stride,
        key_token_stride,
        key_head_stride,
        rotary_token_stride,
        num_kv_heads,
        HEAD_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < HEAD_SIZE
        half_size = HEAD_SIZE // 2
        paired_offsets = tl.where(
            offsets < half_size,
            offsets + half_size,
            offsets - half_size,
        )
        cosines = tl.load(
            cosine_ptr + token * rotary_token_stride + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        sines = tl.load(
            sine_ptr + token * rotary_token_stride + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        query_base = token * query_token_stride + head * query_head_stride
        queries = tl.load(query_ptr + query_base + offsets, mask=mask, other=0.0).to(tl.float32)
        paired_queries = tl.load(
            query_ptr + query_base + paired_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        rotated_queries = tl.where(offsets < half_size, -paired_queries, paired_queries)
        tl.store(
            query_ptr + query_base + offsets,
            queries * cosines + rotated_queries * sines,
            mask=mask,
        )

        active_key = head < num_kv_heads
        key_base = token * key_token_stride + head * key_head_stride
        keys = tl.load(
            key_ptr + key_base + offsets,
            mask=active_key & mask,
            other=0.0,
        ).to(tl.float32)
        paired_keys = tl.load(
            key_ptr + key_base + paired_offsets,
            mask=active_key & mask,
            other=0.0,
        ).to(tl.float32)
        rotated_keys = tl.where(offsets < half_size, -paired_keys, paired_keys)
        tl.store(
            key_ptr + key_base + offsets,
            keys * cosines + rotated_keys * sines,
            mask=active_key & mask,
        )


def _can_use_triton(*tensors: Tensor) -> bool:
    return triton is not None and all(
        tensor.device.type == "cuda"
        and tensor.dtype in (torch.float16, torch.bfloat16)
        and tensor.stride(-1) == 1
        for tensor in tensors
    )


def fused_add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Add a residual and normalize it, reusing both inference input buffers."""

    if hidden_states.shape != residual.shape:
        raise ValueError("hidden states and residual must have the same shape")
    if not _can_use_triton(hidden_states, residual, weight):
        residual = hidden_states + residual
        return (
            F.rms_norm(
                residual,
                (residual.shape[-1],),
                weight,
                eps,
            ),
            residual,
        )

    width = hidden_states.shape[-1]
    rows = hidden_states.numel() // width
    block_size = triton.next_power_of_2(width)
    _fused_add_rms_norm_kernel[(rows,)](
        hidden_states,
        residual,
        weight,
        hidden_states.stride(-2),
        width,
        eps,
        BLOCK_SIZE=block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return hidden_states, residual


def silu_and_mul(gate: Tensor, up: Tensor) -> Tensor:
    """Apply SiLU to the gate and multiply it by the up projection."""

    if gate.shape != up.shape:
        raise ValueError("gate and up projections must have the same shape")
    if not _can_use_triton(gate, up):
        return F.silu(gate) * up

    width = gate.shape[-1]
    rows = gate.numel() // width
    block_size = 256
    _silu_and_mul_kernel[(rows, triton.cdiv(width, block_size))](
        gate,
        up,
        gate.stride(-2),
        width,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return gate


def apply_rotary(
    queries: Tensor,
    keys: Tensor,
    cosines: Tensor,
    sines: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply Qwen2 half-rotation in one optional CUDA kernel."""

    if not _can_use_triton(queries, keys, cosines, sines):
        query_first, query_second = queries.chunk(2, dim=-1)
        key_first, key_second = keys.chunk(2, dim=-1)
        rotated_queries = torch.cat((-query_second, query_first), dim=-1)
        rotated_keys = torch.cat((-key_second, key_first), dim=-1)
        cosines = cosines.unsqueeze(1)
        sines = sines.unsqueeze(1)
        return (
            queries * cosines + rotated_queries * sines,
            keys * cosines + rotated_keys * sines,
        )

    token_rows, num_query_heads, head_size = queries.shape
    num_kv_heads = keys.shape[1]
    block_size = triton.next_power_of_2(head_size)
    _rotary_embedding_kernel[(token_rows, num_query_heads)](
        queries,
        keys,
        cosines,
        sines,
        queries.stride(0),
        queries.stride(1),
        keys.stride(0),
        keys.stride(1),
        cosines.stride(0),
        num_kv_heads,
        HEAD_SIZE=head_size,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return queries, keys
