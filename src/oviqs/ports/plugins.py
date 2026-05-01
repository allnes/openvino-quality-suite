from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol


class PluginRegistryPort(Protocol):
    def names(self) -> Iterable[str]: ...

    def get(self, name: str) -> Any: ...
