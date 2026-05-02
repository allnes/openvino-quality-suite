from __future__ import annotations

from pathlib import Path

import yaml

from oviqs.domain.reports import EvaluationReport, ReportRun, ReportSummary
from oviqs.ports.reporting import EvaluationReportWriterPort


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


def write_suite_scaffold_report(
    config: Path,
    out: Path,
    report_writer: EvaluationReportWriterPort,
) -> None:
    report_writer.write(build_suite_scaffold_report(config, out), out)


__all__ = ["build_suite_scaffold_report", "write_suite_scaffold_report"]
