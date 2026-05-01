from __future__ import annotations

from collections.abc import Iterable
from importlib.metadata import EntryPoint, entry_points
from typing import Any


class EntryPointPluginRegistry:
    def __init__(self, group: str) -> None:
        self.group = group

    def names(self) -> Iterable[str]:
        return [entry_point.name for entry_point in self._entry_points()]

    def get(self, name: str) -> Any:
        for entry_point in self._entry_points():
            if entry_point.name == name:
                return entry_point.load()
        raise KeyError(name)

    def _entry_points(self) -> tuple[EntryPoint, ...]:
        selected = entry_points().select(group=self.group)
        return tuple(selected)
