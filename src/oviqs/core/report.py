from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["pass", "warning", "fail", "unknown"]


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
    raw_sample_metrics_uri: str | None = None


def write_report(report: EvaluationReport | dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json") if isinstance(report, EvaluationReport) else report
    payload.setdefault("schema_version", "openvino_llm_quality_v1")
    if not payload.get("metric_references"):
        from oviqs.references import build_report_reference_manifest

        payload["metric_references"] = build_report_reference_manifest(payload)
    path.write_text(_json_dumps(payload), encoding="utf-8")


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
