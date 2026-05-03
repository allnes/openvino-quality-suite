from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from oviqs.domain.reporting.severity import ReportStatus, Severity


@dataclass(frozen=True)
class MetricObservation:
    path: str
    section: str
    name: str
    value: float | int | str | bool | None
    unit: str | None
    status: ReportStatus
    severity: Severity
    reference_id: str | None
    degradation_rule: str | None
    baseline_value: float | int | None = None
    delta_abs: float | None = None
    delta_rel: float | None = None
    threshold: float | None = None
    threshold_rule: str | None = None
    sample_count: int | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        return payload


@dataclass(frozen=True)
class AnalysisFinding:
    id: str
    title: str
    severity: Severity
    category: str
    status: ReportStatus
    evidence_paths: tuple[str, ...]
    impact: str
    recommendation: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_paths"] = list(self.evidence_paths)
        return payload


@dataclass(frozen=True)
class ReportBundle:
    root: str
    report_json: str
    analysis_json: str
    metrics_csv: str
    index_md: str
    dashboard_html: str
    sample_metrics_jsonl: str
    metadata_json: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["AnalysisFinding", "MetricObservation", "ReportBundle"]
