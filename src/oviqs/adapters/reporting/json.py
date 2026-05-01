from __future__ import annotations

from pathlib import Path
from typing import Any

from oviqs.reporting.json_report import load_report, save_report


class JsonReportAdapter:
    def load(self, path: Path) -> dict[str, Any]:
        return load_report(path)

    def write(self, report: dict[str, Any], path: Path) -> None:
        save_report(report, path)
