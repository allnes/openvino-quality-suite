from __future__ import annotations

from oviqs.core.report import write_report
from oviqs.domain.reports.compatibility import REPORT_CONTRACT_VERSION
from oviqs.domain.reports.models import EvaluationReport, ReportRun, ReportSummary
from oviqs.domain.reports.status import Status

__all__ = [
    "EvaluationReport",
    "REPORT_CONTRACT_VERSION",
    "ReportRun",
    "ReportSummary",
    "Status",
    "write_report",
]
