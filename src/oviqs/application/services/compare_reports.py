from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from oviqs.application.reporting.comparison_service import ReportComparisonService
from oviqs.domain.gates import evaluate_gates
from oviqs.ports.artifacts import ArtifactStorePort
from oviqs.ports.reporting import ReportIOPort


def compare_reports(
    baseline: Path,
    current: Path,
    out: Path,
    report_io: ReportIOPort,
    artifact_store: ArtifactStorePort,
    gates: Path | None = None,
) -> dict[str, Any]:
    base_report = report_io.load(baseline)
    cur_report = report_io.load(current)
    comparison = {
        "baseline": base_report.get("run", {}).get("id"),
        "current": cur_report.get("run", {}).get("id"),
        "summary": cur_report.get("summary", {}),
    }
    evaluated_gates = None
    if gates:
        gate_payload = yaml.safe_load(gates.read_text(encoding="utf-8"))
        evaluated_gates = evaluate_gates(cur_report, gate_payload)
        comparison["gates"] = evaluated_gates
        comparison["summary"]["overall_status"] = evaluated_gates["overall_status"]
    reporting_comparison = ReportComparisonService().compare(
        cur_report,
        baseline=base_report,
        gates=evaluated_gates,
    )
    comparison["reporting_comparison"] = reporting_comparison.to_dict()
    artifact_store.write_text(
        out,
        json.dumps(comparison, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    return comparison


__all__ = ["compare_reports"]
