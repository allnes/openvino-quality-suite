from __future__ import annotations

from typing import Literal

Severity = Literal["none", "low", "medium", "high", "critical"]
ReportStatus = Literal["pass", "warning", "fail", "unknown"]

SEVERITY_ORDER: dict[Severity, int] = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}
STATUS_ORDER: dict[ReportStatus, int] = {
    "pass": 0,
    "unknown": 1,
    "warning": 2,
    "fail": 3,
}


def severity_rank(severity: Severity) -> int:
    return SEVERITY_ORDER[severity]


def status_rank(status: ReportStatus) -> int:
    return STATUS_ORDER[status]


def worst_status(statuses: list[str] | tuple[str, ...]) -> ReportStatus:
    normalized = [status for status in statuses if status in STATUS_ORDER]
    if not normalized:
        return "unknown"
    return max(normalized, key=lambda item: STATUS_ORDER[item])  # type: ignore[return-value]


def status_to_severity(status: str) -> Severity:
    if status == "fail":
        return "high"
    if status == "warning":
        return "medium"
    if status == "unknown":
        return "low"
    return "none"


def finding_severity_sort_key(severity: Severity) -> tuple[int, str]:
    return (-SEVERITY_ORDER[severity], severity)


__all__ = [
    "ReportStatus",
    "SEVERITY_ORDER",
    "STATUS_ORDER",
    "Severity",
    "finding_severity_sort_key",
    "severity_rank",
    "status_rank",
    "status_to_severity",
    "worst_status",
]
