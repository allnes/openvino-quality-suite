from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from oviqs.domain.reporting import ReportAnalysis, ReportBundle, ReportViewModel
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

    def render(self, report: ReportViewModel) -> str: ...


class ReportBundleWriterPort(Protocol):
    def write_bundle(
        self,
        report: dict[str, Any],
        analysis: ReportAnalysis,
        out: Path,
    ) -> ReportBundle: ...


class ReferenceComparisonRendererPort(Protocol):
    def render(
        self,
        comparison: dict[str, Any],
        format_name: str,
    ) -> str: ...
