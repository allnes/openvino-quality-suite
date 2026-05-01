from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from oviqs.domain.reports import EvaluationReport


class ReportReaderPort(Protocol):
    def load(self, path: Path) -> dict[str, Any]: ...


class ReportWriterPort(Protocol):
    def write(self, report: dict[str, Any], path: Path) -> None: ...


class ReportIOPort(ReportReaderPort, ReportWriterPort, Protocol):
    pass


class EvaluationReportWriterPort(Protocol):
    def write(self, report: EvaluationReport | dict[str, Any], path: Path) -> None: ...


class ReportRendererPort(Protocol):
    format_name: str

    def render(self, report: dict[str, Any]) -> str: ...


class ReferenceComparisonWriterPort(Protocol):
    def write(
        self,
        reports: list[str],
        out: Path,
        format_name: str,
        include_all_metrics: bool,
    ) -> None: ...
