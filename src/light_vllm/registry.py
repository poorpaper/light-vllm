from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class RegistryError(LookupError):
    """注册或查找组件失败时抛出。"""


class Registry(Generic[T]):
    """保存名称和组件的对应关系，避免大量 if/elif。"""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, component: T, *, replace: bool = False) -> None:
        if not name:
            raise RegistryError(f"{self._kind} name cannot be empty")
        if name in self._items and not replace:
            raise RegistryError(f"{self._kind} {name!r} is already registered")
        self._items[name] = component

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as error:
            available = ", ".join(self._items) or "<none>"
            raise RegistryError(f"unknown {self._kind} {name!r}; available: {available}") from error

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)
