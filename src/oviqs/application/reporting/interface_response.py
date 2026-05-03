from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oviqs.application.reporting.analysis_service import ReportAnalysisService
from oviqs.application.reporting.schema_normalization import normalize_evaluation_report_contract
from oviqs.domain.reporting.analysis import ReportAnalysis


@dataclass(frozen=True)
class ReportInterfaceResponse:
    run_id: str
    overall_status: str
    report: dict[str, Any]
    analysis: ReportAnalysis
    report_uri: str = ""

    def http_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "overall_status": self.overall_status,
            "report": self.report,
            "analysis": self.analysis.to_dict(),
            "metrics": [metric.to_dict() for metric in self.analysis.metrics],
        }

    def grpc_mapping(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "overall_status": self.overall_status,
            "metrics": [metric.to_dict() for metric in self.analysis.metrics],
            "analysis": self.analysis.to_dict(),
            "report_uri": self.report_uri,
        }


def build_report_interface_response(
    report: dict[str, Any],
    *,
    analysis_service: ReportAnalysisService | None = None,
    report_uri: str = "",
) -> ReportInterfaceResponse:
    normalized = normalize_evaluation_report_contract(report)
    service = analysis_service or ReportAnalysisService()
    analysis = service.analyze(normalized, gates=normalized.get("gates"))
    enriched_report = {**normalized, "analysis": analysis.to_dict()}
    return ReportInterfaceResponse(
        run_id=str(normalized.get("run", {}).get("id", "")),
        overall_status=analysis.summary.overall_status,
        report=enriched_report,
        analysis=analysis,
        report_uri=report_uri,
    )


__all__ = ["ReportInterfaceResponse", "build_report_interface_response"]
