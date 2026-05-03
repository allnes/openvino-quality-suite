from __future__ import annotations

from typing import Any

from oviqs.domain.reporting.analysis import ReportAnalysis, ReportViewModel


def build_report_view_model(report: dict[str, Any], analysis: ReportAnalysis) -> ReportViewModel:
    return ReportViewModel(report=report, analysis=analysis)


__all__ = ["ReportViewModel", "build_report_view_model"]
