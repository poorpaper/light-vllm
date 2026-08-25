"""Serving adapter 共用的生成流关闭边界。"""

from __future__ import annotations

import asyncio
from contextlib import suppress


async def close_stream(events: object) -> None:
    """即使调用方正在取消，也等待底层生成流完成资源清理。"""

    aclose = getattr(events, "aclose", None)
    if aclose is not None:
        pending = asyncio.create_task(aclose())
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError:
            with suppress(Exception):
                await pending
            raise
        return

    close = getattr(events, "close", None)
    if close is not None:
        close()
