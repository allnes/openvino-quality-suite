from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oviqs.application.reporting.schema_normalization import normalize_evaluation_report_contract
from oviqs.domain.references import build_report_reference_manifest
from oviqs.domain.reports import EvaluationReport


class CanonicalReportWriter:
    def write(self, report: EvaluationReport | dict[str, Any], path: Path) -> None:
        payload = (
            report.model_dump(mode="json") if isinstance(report, EvaluationReport) else dict(report)
        )
        payload = normalize_evaluation_report_contract(payload)
        if not payload.get("metric_references"):
            payload["metric_references"] = build_report_reference_manifest(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
