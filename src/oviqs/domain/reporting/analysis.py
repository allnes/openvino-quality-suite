from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from oviqs.domain.reporting.model import AnalysisFinding, MetricObservation
from oviqs.domain.reporting.severity import ReportStatus


@dataclass(frozen=True)
class AnalysisSummary:
    overall_status: ReportStatus
    passed: int
    warning: int
    failed: int
    unknown: int
    finding_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReportAnalysis:
    summary: AnalysisSummary
    findings: tuple[AnalysisFinding, ...] = ()
    metrics: tuple[MetricObservation, ...] = ()
    biggest_regressions: tuple[MetricObservation, ...] = ()
    biggest_improvements: tuple[MetricObservation, ...] = ()
    unknown_metrics: tuple[MetricObservation, ...] = ()
    sample_outliers: tuple[dict[str, Any], ...] = ()
    trend_points: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "findings": [finding.to_dict() for finding in self.findings],
            "metrics": [metric.to_dict() for metric in self.metrics],
            "biggest_regressions": [metric.to_dict() for metric in self.biggest_regressions],
            "biggest_improvements": [metric.to_dict() for metric in self.biggest_improvements],
            "unknown_metrics": [metric.to_dict() for metric in self.unknown_metrics],
            "sample_outliers": list(self.sample_outliers),
            "trend_points": list(self.trend_points),
        }


@dataclass(frozen=True)
class ReportViewModel:
    report: dict[str, Any]
    analysis: ReportAnalysis
    section_order: tuple[str, ...] = field(
        default=(
            "inference_equivalence",
            "likelihood",
            "long_context",
            "generation",
            "rag",
            "agent",
            "serving",
            "performance",
            "reproducibility",
        )
    )


__all__ = ["AnalysisSummary", "ReportAnalysis", "ReportViewModel"]
