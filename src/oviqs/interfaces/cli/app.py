from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from oviqs.application.dto.requests import (
    AgentEvaluationRequest,
    DriftEvaluationRequest,
    GpuSuiteRequest,
    LikelihoodEvaluationRequest,
    LongContextEvaluationRequest,
    RagEvaluationRequest,
    ServingEvaluationRequest,
)
from oviqs.application.services.catalogs import (
    genai_export_commands,
)
from oviqs.application.services.catalogs import (
    list_genai_models as list_genai_model_rows,
)
from oviqs.application.services.catalogs import (
    list_metric_reference_catalog as list_metric_references_from_catalog,
)
from oviqs.application.services.evaluate_agent import build_agent_report
from oviqs.application.services.evaluate_drift import build_drift_report
from oviqs.application.services.evaluate_gpu_suite import build_gpu_suite_report
from oviqs.application.services.evaluate_likelihood import build_likelihood_report
from oviqs.application.services.evaluate_long_context import build_long_context_report
from oviqs.application.services.evaluate_rag import build_rag_report
from oviqs.application.services.evaluate_serving import build_serving_report
from oviqs.application.services.metric_tools import (
    context_gain,
    degradation_slope,
    distractor_sensitivity,
    lost_in_middle_from_json,
)
from oviqs.application.services.reports import (
    compare_reports,
    render_report_to_path,
    write_reference_comparison_table,
    write_suite_scaffold_report,
)
from oviqs.platform.bootstrap import build_default_container

app = typer.Typer(help="OVIQS: OpenVINO Inference Quality Suite")
console = Console()


def _container():
    return build_default_container()


@app.callback()
def main() -> None:
    """Diagnostics for OpenVINO LLM inference quality."""


@app.command("eval-likelihood")
def eval_likelihood(
    model: Annotated[str, typer.Option(help="Model path or id")],
    dataset: Annotated[Path, typer.Option(help="JSONL dataset")],
    out: Annotated[Path, typer.Option(help="Output JSON report")],
    backend: Annotated[
        str, typer.Option(help="dummy, hf, optimum-openvino, openvino-runtime")
    ] = "dummy",
    device: Annotated[str, typer.Option(help="Inference device")] = "CPU",
    window_size: Annotated[int, typer.Option(help="Sliding window size")] = 4096,
    stride: Annotated[int, typer.Option(help="Sliding window stride")] = 1024,
) -> None:
    container = _container()
    report = build_likelihood_report(
        LikelihoodEvaluationRequest(
            model=model,
            dataset=dataset,
            out=out,
            backend=backend,
            device=device,
            window_size=window_size,
            stride=stride,
        ),
        container.runner_factory,
        container.dataset_reader,
    )
    container.report_writer.write(report, out)
    console.print(f"Wrote {out}")


@app.command("eval-drift")
def eval_drift(
    reference: Annotated[str, typer.Option(help="Reference model path/id")],
    current: Annotated[str, typer.Option(help="Current model path/id")],
    dataset: Annotated[Path, typer.Option(help="JSONL dataset")],
    out: Annotated[Path, typer.Option(help="Output JSON report")],
    reference_backend: Annotated[str, typer.Option(help="Reference backend")] = "dummy",
    current_backend: Annotated[str, typer.Option(help="Current backend")] = "dummy",
    reference_device: Annotated[
        str, typer.Option(help="Inference device for reference model")
    ] = "CPU",
    device: Annotated[str, typer.Option(help="Inference device for current model")] = "CPU",
) -> None:
    container = _container()
    try:
        report = build_drift_report(
            DriftEvaluationRequest(
                reference=reference,
                current=current,
                dataset=dataset,
                out=out,
                reference_backend=reference_backend,
                current_backend=current_backend,
                reference_device=reference_device,
                device=device,
            ),
            container.runner_factory,
            container.dataset_reader,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    container.report_writer.write(report, out)
    console.print(f"Wrote {out}")


@app.command("eval-long-context")
def eval_long_context(
    dataset: Annotated[Path, typer.Option(help="JSONL dataset or precomputed metrics")],
    out: Annotated[Path, typer.Option(help="Output JSON report")],
    model: Annotated[str, typer.Option(help="Model path or id")] = "dummy",
    backend: Annotated[str, typer.Option(help="Backend")] = "dummy",
    device: Annotated[str, typer.Option(help="Inference device")] = "CPU",
    lengths: Annotated[
        str, typer.Option(help="Comma-separated context lengths")
    ] = "4096,8192,16384",
    window_size: Annotated[int, typer.Option(help="Sliding-window PPL window")] = 4096,
    stride: Annotated[int, typer.Option(help="Sliding-window PPL stride")] = 1024,
) -> None:
    container = _container()
    try:
        report = build_long_context_report(
            LongContextEvaluationRequest(
                dataset=dataset,
                out=out,
                model=model,
                backend=backend,
                device=device,
                lengths=tuple(int(item) for item in lengths.split(",") if item.strip()),
                window_size=window_size,
                stride=stride,
            ),
            container.runner_factory,
            container.dataset_reader,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    container.report_writer.write(report, out)
    console.print(f"Wrote {out}")


@app.command("eval-serving")
def eval_serving(
    out: Annotated[Path, typer.Option(help="Output JSON report")],
    model: Annotated[str, typer.Option(help="Production model path/id")] = "dummy",
    dataset: Annotated[Path | None, typer.Option(help="Serving JSONL dataset")] = None,
    backend: Annotated[str, typer.Option(help="Logits backend")] = "dummy",
    device: Annotated[str, typer.Option(help="Device")] = "CPU",
) -> None:
    container = _container()
    report = build_serving_report(
        ServingEvaluationRequest(
            out=out,
            model=model,
            dataset=dataset,
            backend=backend,
            device=device,
        ),
        container.runner_factory,
        container.generation_runner_factory,
        container.dataset_reader,
    )
    container.report_writer.write(report, out)
    console.print(f"Wrote {out}")


@app.command("eval-rag")
def eval_rag(
    dataset: Annotated[Path, typer.Option(help="RAG JSONL dataset")],
    out: Annotated[Path, typer.Option(help="Output JSON report")],
    answers: Annotated[Path | None, typer.Option(help="Answers JSONL")] = None,
    scorer: Annotated[str, typer.Option(help="placeholder or ragas")] = "placeholder",
) -> None:
    container = _container()
    report = build_rag_report(
        RagEvaluationRequest(dataset=dataset, out=out, answers=answers, scorer=scorer),
        container.dataset_reader,
        container.rows_reader,
    )
    container.report_writer.write(report, out)
    console.print(f"Wrote {out}")


@app.command("eval-agent")
def eval_agent(
    traces: Annotated[Path, typer.Option(help="Agent traces JSONL")],
    out: Annotated[Path, typer.Option(help="Output JSON report")],
    expected: Annotated[Path | None, typer.Option(help="Expected JSONL")] = None,
) -> None:
    container = _container()
    report = build_agent_report(
        AgentEvaluationRequest(traces=traces, out=out, expected=expected),
        container.rows_reader,
    )
    container.report_writer.write(report, out)
    console.print(f"Wrote {out}")


@app.command("run-suite")
def run_suite(
    config: Annotated[Path, typer.Option(help="Suite YAML config")],
    out: Annotated[Path, typer.Option(help="Output JSON report")],
) -> None:
    container = _container()
    write_suite_scaffold_report(config, out, container.report_writer)
    console.print(f"Wrote {out}")


@app.command("run-gpu-suite")
def run_gpu_suite(
    model: Annotated[str, typer.Option(help="OpenVINO eval/logits model directory")],
    dataset: Annotated[Path, typer.Option(help="Likelihood JSONL dataset")],
    out: Annotated[Path, typer.Option(help="Output JSON report")],
    backend: Annotated[str, typer.Option(help="Logits backend")] = "openvino-runtime",
    device: Annotated[str, typer.Option(help="OpenVINO device for metric verification")] = "GPU",
    window_size: Annotated[int, typer.Option(help="Long-context sliding window")] = 64,
    stride: Annotated[int, typer.Option(help="Long-context sliding stride")] = 32,
    genai_model: Annotated[
        str | None,
        typer.Option(help="Optional OpenVINO GenAI model directory for generation layer"),
    ] = None,
) -> None:
    container = _container()
    try:
        report = build_gpu_suite_report(
            GpuSuiteRequest(
                model=model,
                dataset=dataset,
                out=out,
                backend=backend,
                device=device,
                window_size=window_size,
                stride=stride,
                genai_model=genai_model,
            ),
            container.runner_factory,
            container.generation_runner_factory,
            container.dataset_reader,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    container.report_writer.write(report, out)
    console.print(f"Wrote {out}")


@app.command("compare")
def compare(
    baseline: Annotated[Path, typer.Option(help="Baseline report JSON")],
    current: Annotated[Path, typer.Option(help="Current report JSON")],
    out: Annotated[Path, typer.Option(help="Output comparison JSON")],
    gates: Annotated[Path | None, typer.Option(help="Gate YAML")] = None,
) -> None:
    container = _container()
    compare_reports(baseline, current, out, container.report_io, gates)
    console.print(f"Wrote {out}")


@app.command("render-report")
def render_report(
    report: Annotated[Path, typer.Option(help="Input report JSON")],
    out: Annotated[Path, typer.Option(help="Output rendered report")],
    format: Annotated[str, typer.Option(help="markdown")] = "markdown",
) -> None:
    container = _container()
    try:
        render_report_to_path(report, out, container.report_io, container.markdown_renderer, format)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"Wrote {out}")


@app.command("reference-comparison")
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
    container = _container()
    write_reference_comparison_table(
        report,
        out,
        container.reference_comparison_writer,
        format,
        all_metrics,
    )
    console.print(f"Wrote {out}")


@app.command("metric-long-context")
def metric_long_context(
    position_ppl_json: Annotated[str, typer.Option(help="JSON object with position bucket PPL")],
) -> None:
    console.print(lost_in_middle_from_json(position_ppl_json))


@app.command("list-genai-models")
def list_genai_models(
    config: Annotated[
        Path,
        typer.Option(help="GenAI model matrix YAML"),
    ] = Path("configs/examples/genai_metric_models.yaml"),
    tier: Annotated[str | None, typer.Option(help="Filter by tier")] = None,
    metric: Annotated[str | None, typer.Option(help="Filter by metric")] = None,
    family: Annotated[str | None, typer.Option(help="Filter by family")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    matrix, rows = list_genai_model_rows(config, tier=tier, metric=metric, family=family)
    if json_output:
        console.print_json(
            data=[{"tier": tier_name, **model.model_dump(mode="json")} for tier_name, model in rows]
        )
        return

    table = Table(title="OVIQS GenAI Metric Models")
    table.add_column("Tier")
    table.add_column("Default")
    table.add_column("Model")
    table.add_column("Family")
    table.add_column("Size/Context")
    table.add_column("Metrics")
    for tier_name, model in rows:
        tier_default = matrix.model_matrix[tier_name].default
        marker = "yes" if model.id == tier_default else ""
        size_or_context = model.context or model.size or ""
        table.add_row(
            tier_name,
            marker,
            model.id,
            model.family,
            size_or_context,
            ", ".join(model.metrics),
        )
    console.print(table)


@app.command("list-metric-references")
def list_metric_reference_catalog(
    family: Annotated[str | None, typer.Option(help="Filter by metric family")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    references = list_metric_references_from_catalog(family)
    payload = [reference.to_dict() for reference in references]
    if json_output:
        console.print_json(data=payload)
        return

    table = Table(title="OVIQS Metric References")
    table.add_column("Family")
    table.add_column("Metrics")
    table.add_column("Primary reference")
    table.add_column("Oracle")
    for reference in references:
        table.add_row(
            reference.family,
            ", ".join(reference.metric_names),
            reference.primary_reference,
            reference.oracle,
        )
    console.print(table)


@app.command("genai-export-plan")
def genai_export_plan(
    model: Annotated[
        str,
        typer.Option(help="Model id from the GenAI model matrix"),
    ],
    config: Annotated[
        Path,
        typer.Option(help="GenAI model matrix YAML"),
    ] = Path("configs/examples/genai_metric_models.yaml"),
    output_root: Annotated[
        Path,
        typer.Option(help="Root directory for exported models"),
    ] = Path("models"),
    variant: Annotated[
        list[str] | None,
        typer.Option(help="Export variant; repeat option for multiple variants"),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False,
) -> None:
    commands = genai_export_commands(config, model, output_root, variant)
    if json_output:
        console.print_json(data=[command.model_dump(mode="json") for command in commands])
        return
    for command in commands:
        console.print(command.shell_command)


__all__ = [
    "app",
    "context_gain",
    "degradation_slope",
    "distractor_sensitivity",
]
