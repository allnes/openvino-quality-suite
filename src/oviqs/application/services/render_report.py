from __future__ import annotations

from pathlib import Path

from oviqs.ports.reporting import ReferenceComparisonWriterPort, ReportIOPort, ReportRendererPort


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


__all__ = ["render_report_to_path", "write_reference_comparison_table"]
