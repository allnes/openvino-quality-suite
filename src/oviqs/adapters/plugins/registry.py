from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class InMemoryPluginRegistry:
    def __init__(self, plugins: Mapping[str, Any] | None = None) -> None:
        self._plugins = dict(plugins or {})

    def names(self) -> Iterable[str]:
        return self._plugins.keys()

    def get(self, name: str) -> Any:
        return self._plugins[name]
