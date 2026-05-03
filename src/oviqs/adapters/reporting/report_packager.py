from __future__ import annotations

from pathlib import Path
from typing import Any

from oviqs.application.reporting.package_service import ReportPackageService
from oviqs.domain.reporting import ReportAnalysis, ReportBundle


class ReportPackager:
    def __init__(self, service: ReportPackageService) -> None:
        self.service = service

    def write_bundle(
        self,
        report: dict[str, Any],
        analysis: ReportAnalysis,
        out: Path,
    ) -> ReportBundle:
        return self.service.build(report, out, analysis=analysis)


__all__ = ["ReportPackager"]
