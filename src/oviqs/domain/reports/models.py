from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from oviqs.domain.reports.status import Status


class ReportRun(BaseModel):
    id: str
    suite: str = "openvino_llm_quality_v1"
    model: str | None = None
    reference: str | None = None
    current: str | None = None
    device: str | None = None
    precision: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ReportSummary(BaseModel):
    overall_status: Status = "unknown"
    main_findings: list[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    schema_version: str = "openvino_llm_quality_v1"
    run: ReportRun
    summary: ReportSummary = Field(default_factory=ReportSummary)
    inference_equivalence: dict[str, Any] = Field(default_factory=dict)
    likelihood: dict[str, Any] = Field(default_factory=dict)
    long_context: dict[str, Any] = Field(default_factory=dict)
    robustness: dict[str, Any] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)
    rag: dict[str, Any] = Field(default_factory=dict)
    agent: dict[str, Any] = Field(default_factory=dict)
    serving: dict[str, Any] = Field(default_factory=dict)
    performance: dict[str, Any] = Field(default_factory=dict)
    gates: dict[str, Any] = Field(default_factory=dict)
    metric_references: dict[str, Any] = Field(default_factory=dict)
    reproducibility: dict[str, Any] = Field(default_factory=dict)
    analysis: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    report_metadata: dict[str, Any] = Field(default_factory=dict)
    ui_hints: dict[str, Any] = Field(default_factory=dict)
    sample_metrics_summary: dict[str, Any] = Field(default_factory=dict)
    raw_sample_metrics_uri: str | None = None


__all__ = ["EvaluationReport", "ReportRun", "ReportSummary"]
