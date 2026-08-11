from __future__ import annotations

from typing import Protocol


class ExecutionError(RuntimeError):
    """执行一次模型计算失败时抛出。"""


class ExecutionNotReadyError(ExecutionError):
    """模型执行器还没准备好时抛出。"""


class TokenExecutor(Protocol):
    """根据当前 token 序列计算下一个 token。"""

    @property
    def ready(self) -> bool: ...

    def next_token(self, token_ids: tuple[int, ...]) -> int: ...
