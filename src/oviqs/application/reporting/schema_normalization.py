from __future__ import annotations

from copy import deepcopy
from typing import Any

from oviqs.domain.reports import REPORT_CONTRACT_VERSION

SUPPORTED_REPORT_VERSIONS = {REPORT_CONTRACT_VERSION}


class UnsupportedReportSchemaVersion(ValueError):
    def __init__(self, version: Any) -> None:
        super().__init__(f"Unsupported EvaluationReport schema_version: {version!r}")
        self.version = version


def normalize_evaluation_report_contract(report: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(report)
    version = payload.get("schema_version")
    if version != REPORT_CONTRACT_VERSION:
        raise UnsupportedReportSchemaVersion(version)
    return _ensure_optional_current_sections(payload)


def _ensure_optional_current_sections(report: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "metric_references",
        "reproducibility",
        "analysis",
        "artifacts",
        "report_metadata",
        "ui_hints",
        "sample_metrics_summary",
    ):
        if not isinstance(report.get(key), dict):
            report[key] = {}
    return report


__all__ = [
    "SUPPORTED_REPORT_VERSIONS",
    "UnsupportedReportSchemaVersion",
    "normalize_evaluation_report_contract",
]
