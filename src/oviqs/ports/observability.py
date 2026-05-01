from __future__ import annotations

from typing import Any, Protocol


class MetricsSinkPort(Protocol):
    def record(
        self, name: str, value: float, _attributes: dict[str, Any] | None = None
    ) -> None: ...


class TraceSinkPort(Protocol):
    def event(self, name: str, _attributes: dict[str, Any] | None = None) -> None: ...
