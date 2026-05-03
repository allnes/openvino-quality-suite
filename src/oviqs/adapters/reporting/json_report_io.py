from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oviqs.application.reporting.schema_normalization import normalize_evaluation_report_contract


class JsonReportAdapter:
    def load(self, path: Path) -> dict[str, Any]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("EvaluationReport JSON must be an object")
        return normalize_evaluation_report_contract(payload)

    def write(self, report: dict[str, Any], path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = normalize_evaluation_report_contract(report)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )


__all__ = ["JsonReportAdapter"]
