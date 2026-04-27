from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T) -> None:
        if name in self._items:
            raise ValueError(f"Duplicate registry entry: {name}")
        self._items[name] = item

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as exc:
            raise KeyError(f"Unknown registry entry: {name}") from exc

    def decorator(self, name: str) -> Callable[[T], T]:
        def _inner(item: T) -> T:
            self.register(name, item)
            return item

        return _inner
