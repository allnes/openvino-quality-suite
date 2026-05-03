from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oviqs.application.reporting.analysis_service import ReportAnalysisService
from oviqs.domain.reporting import ReportAnalysis


@dataclass(frozen=True)
class GeneratedReport:
    report: dict[str, Any]
    analysis: ReportAnalysis


class ReportGenerationService:
    def __init__(self, analysis_service: ReportAnalysisService) -> None:
        self.analysis_service = analysis_service

    def enrich(
        self,
        report: dict[str, Any],
        *,
        baseline: dict[str, Any] | None = None,
        gates: dict[str, Any] | None = None,
        analysis: ReportAnalysis | None = None,
    ) -> GeneratedReport:
        analysis = analysis or self.analysis_service.analyze(report, baseline=baseline, gates=gates)
        enriched = dict(report)
        enriched["analysis"] = analysis.to_dict()
        enriched.setdefault(
            "report_metadata",
            {
                "format": "evaluation_report",
                "analysis_version": "openvino_llm_quality_reporting_v1",
            },
        )
        return GeneratedReport(report=enriched, analysis=analysis)


__all__ = ["GeneratedReport", "ReportGenerationService"]
