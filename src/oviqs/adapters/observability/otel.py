from __future__ import annotations

from typing import Any


class OTelTraceSink:
    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    def event(self, name: str, _attributes: dict[str, Any] | None = None) -> None:
        span = self._tracer.start_span(name)
        for key, value in (_attributes or {}).items():
            span.set_attribute(key, value)
        span.end()
