from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from oviqs.domain.gates import evaluate_gates
from oviqs.ports.reporting import ReportIOPort


def compare_reports(
    baseline: Path,
    current: Path,
    out: Path,
    report_io: ReportIOPort,
    gates: Path | None = None,
) -> dict[str, Any]:
    base_report = report_io.load(baseline)
    cur_report = report_io.load(current)
    comparison = {
        "baseline": base_report.get("run", {}).get("id"),
        "current": cur_report.get("run", {}).get("id"),
        "summary": cur_report.get("summary", {}),
    }
    if gates:
        gate_payload = yaml.safe_load(gates.read_text(encoding="utf-8"))
        comparison["gates"] = evaluate_gates(cur_report, gate_payload)
        comparison["summary"]["overall_status"] = comparison["gates"]["overall_status"]
    report_io.write(comparison, out)
    return comparison


__all__ = ["compare_reports"]
