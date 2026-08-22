"""单个模型 step 内复用的 CUDA 输入暂存区。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


class SingleStepCudaStagingBuffer:
    """把多组小整数合并成一次异步 H2D。

    返回的 view 只在下一次 ``copy_groups`` 前有效。当前 Worker 每次只允许一个
    model step 在途；未来若增加多批并行，应为每个在途槽位各持有一个实例。
    """

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        if device.type != "cuda":
            raise ValueError("CUDA staging requires a CUDA device")
        self._device = device
        self._dtype = dtype
        self._host_buffer: Tensor | None = None
        self._device_buffer: Tensor | None = None
        self._host_copy_done: torch.cuda.Event | None = None

    def copy_groups(self, *groups: Sequence[int]) -> tuple[Tensor, ...]:
        lengths = tuple(len(group) for group in groups)
        total = sum(lengths)
        if total == 0:
            raise ValueError("CUDA staging requires at least one value")
        self._wait_until_host_reusable()
        self._ensure_capacity(total)
        assert self._host_buffer is not None
        assert self._device_buffer is not None

        values = [value for group in groups for value in group]
        self._host_buffer[:total].copy_(torch.tensor(values, dtype=self._dtype))
        self._device_buffer[:total].copy_(self._host_buffer[:total], non_blocking=True)
        if self._host_copy_done is None:
            self._host_copy_done = torch.cuda.Event(blocking=True)
        self._host_copy_done.record(torch.cuda.current_stream(self._device))

        views: list[Tensor] = []
        start = 0
        for length in lengths:
            views.append(self._device_buffer[start : start + length])
            start += length
        return tuple(views)

    def _wait_until_host_reusable(self) -> None:
        if self._host_copy_done is not None:
            # 只等上一轮 H2D 读完 pinned buffer，不等待后续模型计算。
            self._host_copy_done.synchronize()

    def _ensure_capacity(self, required: int) -> None:
        if self._device_buffer is not None and self._device_buffer.numel() >= required:
            return
        capacity = 1 << (required - 1).bit_length()
        self._host_buffer = torch.empty(
            capacity,
            dtype=self._dtype,
            device="cpu",
            pin_memory=True,
        )
        self._device_buffer = torch.empty(
            capacity,
            dtype=self._dtype,
            device=self._device,
        )
