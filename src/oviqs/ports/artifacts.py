from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from oviqs.domain.reporting import MetricObservation


class ArtifactStorePort(Protocol):
    def ensure_dir(self, path: Path) -> None: ...
    def write_text(self, path: Path, content: str) -> None: ...


class SampleMetricsStorePort(Protocol):
    def render_metrics(self, metrics: list[dict[str, Any]]) -> str: ...
    def write_metrics(self, metrics: list[dict[str, Any]], path: Path) -> None: ...


class MetricTableWriterPort(Protocol):
    def render(self, metrics: list[MetricObservation]) -> str: ...
    def write(self, metrics: list[MetricObservation], path: Path) -> None: ...


__all__ = ["ArtifactStorePort", "MetricTableWriterPort", "SampleMetricsStorePort"]
