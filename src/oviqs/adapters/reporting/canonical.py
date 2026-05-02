from __future__ import annotations

from pathlib import Path
from typing import Any

from oviqs.domain.reports import EvaluationReport, write_report


class CanonicalReportWriter:
    def write(self, report: EvaluationReport | dict[str, Any], path: Path) -> None:
        write_report(report, path)
