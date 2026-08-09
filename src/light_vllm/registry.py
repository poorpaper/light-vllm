from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class RegistryError(LookupError):
    """Raised when a component cannot be registered or resolved."""


class Registry(Generic[T]):
    """A small explicit map used instead of feature-dispatch condition trees."""

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
