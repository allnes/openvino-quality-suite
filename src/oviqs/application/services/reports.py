from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from oviqs.domain.gates import evaluate_gates
from oviqs.domain.reports import EvaluationReport, ReportRun, ReportSummary
from oviqs.ports.reporting import (
    EvaluationReportWriterPort,
    ReferenceComparisonWriterPort,
    ReportIOPort,
    ReportRendererPort,
)


def build_suite_scaffold_report(config: Path, out: Path) -> EvaluationReport:
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    run = payload.get("run", {})
    return EvaluationReport(
        run=ReportRun(
            id=run.get("id", out.stem),
            suite=run.get("suite", "openvino_llm_quality_v1"),
        ),
        summary=ReportSummary(
            overall_status="unknown",
            main_findings=["Suite orchestration scaffold is ready."],
        ),
    )


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


def render_report_to_path(
    report: Path,
    out: Path,
    report_reader: ReportIOPort,
    renderer: ReportRendererPort,
    format_name: str = "markdown",
) -> None:
    payload = report_reader.load(report)
    if format_name != renderer.format_name:
        raise ValueError("Only markdown rendering is implemented in v0.1.0")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(renderer.render(payload), encoding="utf-8")


def write_reference_comparison_table(
    reports: list[str],
    out: Path,
    comparison_writer: ReferenceComparisonWriterPort,
    format_name: str = "markdown",
    include_all_metrics: bool = False,
) -> None:
    comparison_writer.write(reports, out, format_name, include_all_metrics)


def write_suite_scaffold_report(
    config: Path,
    out: Path,
    report_writer: EvaluationReportWriterPort,
) -> None:
    report_writer.write(build_suite_scaffold_report(config, out), out)
