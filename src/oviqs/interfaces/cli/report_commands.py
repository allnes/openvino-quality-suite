from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from oviqs.platform.bootstrap import build_default_container

report_app = typer.Typer(help="Build, analyze and render OVIQS report bundles.")


@report_app.command("build")
def build_report_bundle(
    report: Annotated[Path, typer.Option(help="Input EvaluationReport JSON")],
    out: Annotated[Path, typer.Option(help="Output report bundle directory")],
    baseline: Annotated[Path | None, typer.Option(help="Optional baseline report JSON")] = None,
    gates: Annotated[Path | None, typer.Option(help="Optional gates result JSON")] = None,
    trend_history: Annotated[
        Path | None,
        typer.Option(help="Optional report-history JSONL used as trend baseline"),
    ] = None,
    format: Annotated[str, typer.Option(help="all")] = "all",
) -> None:
    workflow = build_default_container().report_workflow_service(trend_history)
    try:
        bundle = workflow.build_bundle(
            report,
            out,
            baseline_path=baseline,
            gates_path=gates,
            format_name=format,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Wrote {bundle.root}")


@report_app.command("analyze")
def analyze_report(
    report: Annotated[Path, typer.Option(help="Input EvaluationReport JSON")],
    out: Annotated[Path, typer.Option(help="Output analysis JSON")],
    baseline: Annotated[Path | None, typer.Option(help="Optional baseline report JSON")] = None,
    gates: Annotated[Path | None, typer.Option(help="Optional gates YAML")] = None,
    trend_history: Annotated[
        Path | None,
        typer.Option(help="Optional report-history JSONL used as trend baseline"),
    ] = None,
) -> None:
    workflow = build_default_container().report_workflow_service(trend_history)
    try:
        workflow.analyze_report(report, out, baseline_path=baseline, gates_path=gates)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Wrote {out}")


@report_app.command("render")
def render_report_bundle(
    bundle: Annotated[Path, typer.Option(help="Report bundle directory")],
    out: Annotated[Path, typer.Option(help="Output rendered artifact")],
    format: Annotated[str, typer.Option(help="markdown or html-dashboard")] = "markdown",
) -> None:
    workflow = build_default_container().report_workflow_service()
    try:
        workflow.render_bundle(bundle, out, format_name=format)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Wrote {out}")


@report_app.command("metrics-table")
def metrics_table(
    report: Annotated[Path, typer.Option(help="Input EvaluationReport JSON")],
    out: Annotated[Path, typer.Option(help="Output metrics CSV")],
    trend_history: Annotated[
        Path | None,
        typer.Option(help="Optional report-history JSONL used as trend baseline"),
    ] = None,
) -> None:
    workflow = build_default_container().report_workflow_service(trend_history)
    try:
        workflow.write_metrics_table(report, out)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Wrote {out}")


@report_app.command("validate")
def validate_report(
    report: Annotated[Path, typer.Option(help="Input EvaluationReport JSON")],
) -> None:
    try:
        errors = build_default_container().report_workflow_service().validate_report(report)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if errors:
        raise typer.BadParameter("; ".join(errors))
    typer.echo("Report is structurally valid")


@report_app.command("reference-comparison")
def reference_comparison(
    report: Annotated[
        list[str],
        typer.Option(
            "--report",
            help="Report JSON path, optionally LABEL=PATH. Repeat for multiple reports.",
        ),
    ],
    out: Annotated[Path, typer.Option(help="Output comparison table")],
    format: Annotated[
        str,
        typer.Option(help="markdown, markdown-transposed, html-dashboard, csv or json"),
    ] = "markdown",
    all_metrics: Annotated[
        bool,
        typer.Option("--all-metrics", help="Include every metric listed in report coverage"),
    ] = False,
) -> None:
    if not report:
        raise typer.BadParameter("At least one --report is required")
    workflow = build_default_container().report_workflow_service()
    try:
        workflow.write_reference_comparison(
            report,
            out,
            format_name=format,
            include_all_metrics=all_metrics,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Wrote {out}")


__all__ = ["report_app"]
