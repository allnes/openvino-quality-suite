from __future__ import annotations

from typing import Any


class NullMetricsSink:
    def record(
        self,
        name: str,
        value: float,
        _attributes: dict[str, Any] | None = None,
    ) -> None:
        return None


class NullTraceSink:
    def event(self, name: str, _attributes: dict[str, Any] | None = None) -> None:
        return None
