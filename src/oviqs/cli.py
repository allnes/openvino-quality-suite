from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import yaml
from rich.console import Console
from rich.table import Table

from oviqs.aggregation.buckets import aggregate_position_bucketed_ppl
from oviqs.aggregation.gates import evaluate_gates
from oviqs.core.report import EvaluationReport, ReportRun, ReportSummary, write_report
from oviqs.core.sample import EvalSample
from oviqs.core.trace import AgentTrace, TraceStep
from oviqs.datasets.jsonl import load_jsonl_samples, read_jsonl
from oviqs.metrics.agent import (
    agent_state_drift as compute_agent_state_drift,
)
from oviqs.metrics.agent import (
    observation_grounding_score as compute_observation_grounding_score,
)
from oviqs.metrics.agent import (
    observation_grounding_score_placeholder,
    policy_violation_rate,
    recovery_after_tool_error,
    redundant_tool_call_rate,
    task_completion,
    tool_call_validity,
)
from oviqs.metrics.distribution_drift import (
    aggregate_drift,
    distribution_drift,
    top1_changed_rate,
    topk_overlap,
)
from oviqs.metrics.generation import json_validity, ngram_repetition_rate
from oviqs.metrics.likelihood import nll_ppl_from_logits, sliding_window_ppl
from oviqs.metrics.long_context import (
    authoritative_margin,
    conflict_entropy,
    conflict_sensitivity,
    context_gain,
    degradation_slope,
    distractor_sensitivity,
    lost_in_middle_score_from_ppl,
)
from oviqs.metrics.rag import (
    citation_metrics,
    context_precision,
    context_recall,
    distractor_ratio,
    evidence_coverage,
    rule_based_faithfulness,
    supported_claim_ratio_placeholder,
)
from oviqs.metrics.serving import batch_invariance_drift, kv_cache_drift_interface
from oviqs.models.genai_matrix import export_plan, load_genai_model_matrix
from oviqs.references import list_metric_references, references_for_family
from oviqs.reporting.json_report import load_report, save_report
from oviqs.reporting.markdown_report import render_markdown_report
from oviqs.reporting.reference_comparison import (
    build_reference_comparison,
    parse_report_inputs,
    write_reference_comparison,
)
from oviqs.runners.dummy import DummyLogitsRunner

app = typer.Typer(help="OVIQS: OpenVINO Inference Quality Suite")
console = Console()


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
    runner = _build_logits_runner(backend, model, device)
    sample_metrics = []
    nll_sum = 0.0
    token_count = 0
    for sample in load_jsonl_samples(dataset):
        text = sample.text or " ".join(
            part for part in [sample.context, sample.prompt, sample.target] if part
        )
        encoded = _encode_for_runner(runner, text)
        result = (
            sliding_window_ppl(
                runner, encoded["input_ids"], encoded.get("attention_mask"), window_size, stride
            )
            if encoded["input_ids"].shape[1] > window_size
            else nll_ppl_from_logits(
                runner.forward_logits(encoded["input_ids"], encoded.get("attention_mask")),
                encoded["input_ids"],
                encoded.get("attention_mask"),
            )
        )
        sample_metrics.append({"id": sample.id, **result})
        nll_sum += float(result["nll"]) * int(result["num_tokens"])
        token_count += int(result["num_tokens"])
    mean_nll = nll_sum / max(token_count, 1)
    report = EvaluationReport(
        run=ReportRun(id=out.stem, model=model, current=backend, device=device),
        summary=ReportSummary(overall_status="pass"),
        likelihood={
            "nll": mean_nll,
            "perplexity": float(np.exp(mean_nll)),
            "num_tokens": token_count,
            "samples": sample_metrics,
        },
    )
    write_report(report, out)
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
    ref_runner = _build_logits_runner(reference_backend, reference, reference_device)
    cur_runner = _build_logits_runner(current_backend, current, device)
    sample_metrics = []
    aggregates = []
    for sample in load_jsonl_samples(dataset):
        text = sample.text or " ".join(
            part for part in [sample.context, sample.prompt, sample.target] if part
        )
        encoded = _encode_for_runner(ref_runner, text)
        ref_logits = ref_runner.forward_logits(encoded["input_ids"], encoded.get("attention_mask"))[
            :, :-1, :
        ]
        cur_logits = cur_runner.forward_logits(encoded["input_ids"], encoded.get("attention_mask"))[
            :, :-1, :
        ]
        if ref_logits.shape != cur_logits.shape:
            message = (
                f"Logits shape mismatch for sample {sample.id}: "
                f"{ref_logits.shape} vs {cur_logits.shape}"
            )
            raise typer.BadParameter(message)
        agg = aggregate_drift(distribution_drift(ref_logits, cur_logits))
        sample_metrics.append({"id": sample.id, **agg})
        aggregates.append(agg)
    report_agg = (
        {key: float(np.mean([item[key] for item in aggregates])) for key in aggregates[0]}
        if aggregates
        else {}
    )
    report = EvaluationReport(
        run=ReportRun(
            id=out.stem, reference=reference_backend, current=current_backend, device=device
        ),
        summary=ReportSummary(overall_status="pass"),
        inference_equivalence={**report_agg, "samples": sample_metrics},
    )
    write_report(report, out)
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
    runner = _build_logits_runner(backend, model, device)
    samples = load_jsonl_samples(dataset)
    if not samples:
        raise typer.BadParameter("long-context dataset must contain at least one sample")
    requested_lengths = [int(item) for item in lengths.split(",") if item.strip()]
    section = _compute_eval_long_context_section(
        runner,
        samples,
        requested_lengths,
        window_size,
        stride,
    )
    statuses = [section.get("status", "unknown")]
    report = EvaluationReport(
        run=ReportRun(id=out.stem, model=model, current=backend, device=device),
        summary=ReportSummary(
            overall_status=_overall_status(statuses),
            main_findings=["Long-context metrics were computed from the provided JSONL samples."],
        ),
        long_context=section,
    )
    write_report(report, out)
    console.print(f"Wrote {out}")


@app.command("eval-serving")
def eval_serving(
    out: Annotated[Path, typer.Option(help="Output JSON report")],
    model: Annotated[str, typer.Option(help="Production model path/id")] = "dummy",
    dataset: Annotated[Path | None, typer.Option(help="Serving JSONL dataset")] = None,
    backend: Annotated[str, typer.Option(help="Logits backend")] = "dummy",
    device: Annotated[str, typer.Option(help="Device")] = "CPU",
) -> None:
    samples = load_jsonl_samples(dataset) if dataset else _default_serving_samples()
    runner = _build_logits_runner(backend, model, device)
    serving = _safe_section("serving", lambda: _compute_serving_section(runner, samples))
    generation_runner = runner
    if backend == "dummy":
        from oviqs.runners.dummy import DummyGenerationRunner

        generation_runner = DummyGenerationRunner()
    generation = _compute_serving_generation_section(generation_runner, samples)
    report = EvaluationReport(
        run=ReportRun(id=out.stem, model=model, current=backend, device=device),
        summary=ReportSummary(overall_status=_overall_status([serving.get("status", "unknown")])),
        serving={**serving, **generation},
    )
    write_report(report, out)
    console.print(f"Wrote {out}")


@app.command("eval-rag")
def eval_rag(
    dataset: Annotated[Path, typer.Option(help="RAG JSONL dataset")],
    out: Annotated[Path, typer.Option(help="Output JSON report")],
    answers: Annotated[Path | None, typer.Option(help="Answers JSONL")] = None,
    scorer: Annotated[str, typer.Option(help="placeholder or ragas")] = "placeholder",
) -> None:
    samples = load_jsonl_samples(dataset)
    answer_rows = _read_optional_rows(answers)
    rag = _compute_eval_rag_section(samples, answer_rows, scorer)
    report = EvaluationReport(
        run=ReportRun(id=out.stem),
        summary=ReportSummary(overall_status=_overall_status([rag.get("status", "unknown")])),
        rag=rag,
    )
    write_report(report, out)
    console.print(f"Wrote {out}")


@app.command("eval-agent")
def eval_agent(
    traces: Annotated[Path, typer.Option(help="Agent traces JSONL")],
    out: Annotated[Path, typer.Option(help="Output JSON report")],
    expected: Annotated[Path | None, typer.Option(help="Expected JSONL")] = None,
) -> None:
    trace_rows = _read_rows(traces)
    expected_rows = _read_optional_rows(expected)
    agent = _compute_eval_agent_section(trace_rows, expected_rows)
    report = EvaluationReport(
        run=ReportRun(id=out.stem),
        summary=ReportSummary(overall_status=_overall_status([agent.get("status", "unknown")])),
        agent=agent,
    )
    write_report(report, out)
    console.print(f"Wrote {out}")


@app.command("run-suite")
def run_suite(
    config: Annotated[Path, typer.Option(help="Suite YAML config")],
    out: Annotated[Path, typer.Option(help="Output JSON report")],
) -> None:
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    run = payload.get("run", {})
    report = EvaluationReport(
        run=ReportRun(
            id=run.get("id", out.stem), suite=run.get("suite", "openvino_llm_quality_v1")
        ),
        summary=ReportSummary(
            overall_status="unknown", main_findings=["Suite orchestration scaffold is ready."]
        ),
    )
    write_report(report, out)
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
    runner = _build_logits_runner(backend, model, device)
    samples = load_jsonl_samples(dataset)
    if not samples:
        raise typer.BadParameter("GPU suite dataset must contain at least one sample")

    likelihood = _safe_section("likelihood", lambda: _compute_likelihood_section(runner, samples))
    inference_equivalence = _safe_section(
        "inference_equivalence",
        lambda: _compute_self_drift_section(runner, samples),
    )
    long_context = _safe_section(
        "long_context",
        lambda: _compute_long_context_section(runner, samples[0], window_size, stride),
    )
    serving = _safe_section("serving", lambda: _compute_serving_section(runner, samples))
    generation = _safe_section(
        "generation",
        lambda: _compute_generation_section(genai_model, device),
    )
    rag = _safe_section("rag", _compute_rag_section)
    agent = _safe_section("agent", _compute_agent_section)
    section_statuses = [
        section.get("status", "unknown")
        for section in [
            likelihood,
            inference_equivalence,
            long_context,
            serving,
            generation,
            rag,
            agent,
        ]
    ]

    report = EvaluationReport(
        run=ReportRun(
            id=out.stem,
            suite="openvino_llm_quality_v1_gpu",
            model=model,
            current=backend,
            device=device,
        ),
        summary=ReportSummary(
            overall_status=_overall_status(section_statuses),
            main_findings=_gpu_suite_findings(section_statuses),
        ),
        inference_equivalence=inference_equivalence,
        likelihood=likelihood,
        long_context=long_context,
        generation=generation,
        rag=rag,
        agent=agent,
        serving=serving,
    )
    write_report(report, out)
    console.print(f"Wrote {out}")


@app.command("compare")
def compare(
    baseline: Annotated[Path, typer.Option(help="Baseline report JSON")],
    current: Annotated[Path, typer.Option(help="Current report JSON")],
    out: Annotated[Path, typer.Option(help="Output comparison JSON")],
    gates: Annotated[Path | None, typer.Option(help="Gate YAML")] = None,
) -> None:
    base_report = load_report(baseline)
    cur_report = load_report(current)
    comparison = {
        "baseline": base_report.get("run", {}).get("id"),
        "current": cur_report.get("run", {}).get("id"),
        "summary": cur_report.get("summary", {}),
    }
    if gates:
        gate_payload = yaml.safe_load(gates.read_text(encoding="utf-8"))
        comparison["gates"] = evaluate_gates(cur_report, gate_payload)
        comparison["summary"]["overall_status"] = comparison["gates"]["overall_status"]
    save_report(comparison, out)
    console.print(f"Wrote {out}")


@app.command("render-report")
def render_report(
    report: Annotated[Path, typer.Option(help="Input report JSON")],
    out: Annotated[Path, typer.Option(help="Output rendered report")],
    format: Annotated[str, typer.Option(help="markdown")] = "markdown",
) -> None:
    payload = load_report(report)
    if format != "markdown":
        raise typer.BadParameter("Only markdown rendering is implemented in v0.1.0")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown_report(payload), encoding="utf-8")
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
    comparison = build_reference_comparison(
        parse_report_inputs(report),
        include_all_metrics=all_metrics,
    )
    write_reference_comparison(comparison, out, format)
    console.print(f"Wrote {out}")


@app.command("metric-long-context")
def metric_long_context(
    position_ppl_json: Annotated[str, typer.Option(help="JSON object with position bucket PPL")],
) -> None:
    values = json.loads(position_ppl_json)
    console.print({"lost_in_middle_score": lost_in_middle_score_from_ppl(values)})


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
    matrix = load_genai_model_matrix(config)
    rows = matrix.list_models(tier=tier, metric=metric, family=family)
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
    references = references_for_family(family) if family else list_metric_references()
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
    matrix = load_genai_model_matrix(config)
    commands = export_plan(matrix, model_id=model, output_root=output_root, variants=variant)
    if json_output:
        console.print_json(data=[command.model_dump(mode="json") for command in commands])
        return
    for command in commands:
        console.print(command.shell_command)


def _build_logits_runner(backend: str, model: str, device: str):
    if backend == "dummy":
        return DummyLogitsRunner()
    if backend == "hf":
        from oviqs.runners.hf_reference import HFReferenceRunner

        return HFReferenceRunner(model, device=device.lower())
    if backend == "optimum-openvino":
        from oviqs.runners.optimum_openvino import OptimumOVLogitsRunner

        return OptimumOVLogitsRunner(model, device=device)
    if backend == "openvino-runtime":
        from oviqs.runners.openvino_runtime import OVRuntimeLogitsRunner

        return OVRuntimeLogitsRunner(model, device=device)
    raise typer.BadParameter(f"Unsupported backend: {backend}")


def _encode_for_runner(runner, text: str) -> dict[str, np.ndarray]:
    if hasattr(runner, "encode"):
        encoded = runner.encode(text)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask") if hasattr(encoded, "get") else None
        if hasattr(input_ids, "detach"):
            input_ids = input_ids.detach().cpu().numpy()
        if hasattr(attention_mask, "detach"):
            attention_mask = attention_mask.detach().cpu().numpy()
        return {
            "input_ids": np.asarray(input_ids),
            "attention_mask": None if attention_mask is None else np.asarray(attention_mask),
        }
    ids = [1] + [abs(hash(token)) % 15 for token in text.split()] + [2]
    return {
        "input_ids": np.asarray([ids], dtype=np.int64),
        "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
    }


def _sample_text(sample) -> str:
    return sample.text or " ".join(
        part for part in [sample.context, sample.prompt, sample.target] if part
    )


def _read_rows(path: Path) -> list[dict]:
    return read_jsonl(path)


def _read_optional_rows(path: Path | None) -> list[dict]:
    return read_jsonl(path) if path else []


def _default_serving_samples() -> list[EvalSample]:
    return [
        EvalSample(id="serving_1", task_type="serving", text="openvino serving metrics"),
        EvalSample(id="serving_2", task_type="serving", text="batch invariance metrics"),
    ]


def _compute_eval_long_context_section(
    runner,
    samples,
    lengths: list[int],
    window_size: int,
    stride: int,
) -> dict:
    sample_reports = []
    position_bucket_inputs = []
    length_to_quality: dict[int, float] = {}
    nll_by_context: dict[str, float] = {}

    for sample in samples:
        text = _sample_text(sample)
        encoded = _encode_for_runner(runner, text)
        base_result = nll_ppl_from_logits(
            runner.forward_logits(encoded["input_ids"], encoded.get("attention_mask")),
            encoded["input_ids"],
            encoded.get("attention_mask"),
        )
        nll_by_context.setdefault("0k", float(base_result["nll"]))
        sample_report = {"id": sample.id, "baseline": base_result}

        sample_lengths = {}
        for length in lengths:
            long_text = _expand_text_to_token_length(runner, text, length)
            long_encoded = _encode_for_runner(runner, long_text)
            result = sliding_window_ppl(
                runner,
                long_encoded["input_ids"],
                long_encoded.get("attention_mask"),
                window_size=window_size,
                stride=stride,
            )
            sample_lengths[str(length)] = {
                "nll": result["nll"],
                "perplexity": result["perplexity"],
                "num_tokens": result["num_tokens"],
            }
            length_to_quality[length] = -float(result["nll"])
            nll_by_context[f"{length}"] = float(result["nll"])
            position_bucket_inputs.extend(result["per_token"])

        sample_report["lengths"] = sample_lengths
        sample_reports.append(sample_report)

    seq_len = max((int(item["absolute_pos"]) for item in position_bucket_inputs), default=0) + 1
    position_buckets = (
        aggregate_position_bucketed_ppl(position_bucket_inputs, seq_len=seq_len)
        if position_bucket_inputs
        else {}
    )
    position_ppl = {bucket: values["ppl"] for bucket, values in position_buckets.items()}
    lim_score = (
        lost_in_middle_score_from_ppl(position_ppl)
        if {"0_10", "30_50", "50_70", "90_100"}.issubset(position_ppl)
        else None
    )
    gains = context_gain(nll_by_context, baseline_key="0k") if len(nll_by_context) > 1 else {}
    slope = degradation_slope(length_to_quality) if len(length_to_quality) > 1 else None

    robustness = _compute_long_context_robustness(samples)
    return {
        "samples": sample_reports,
        "context_gain": gains,
        "degradation_slope": slope,
        "position_bucketed_ppl": position_buckets,
        "lost_in_middle_score": lim_score,
        **robustness,
        "status": "pass" if lim_score is not None else "warning",
        "warnings": [] if lim_score is not None else ["Not enough position buckets for LITM score"],
    }


def _expand_text_to_token_length(runner, text: str, target_length: int) -> str:
    if not text.strip():
        raise ValueError("Cannot expand empty text for long-context metrics")
    parts = [text]
    while True:
        expanded = " ".join(parts)
        encoded = _encode_for_runner(runner, expanded)
        if encoded["input_ids"].shape[1] >= target_length:
            return expanded
        parts.append(text)


def _compute_long_context_robustness(samples) -> dict:
    clean_nll = None
    distracted_nll = None
    conflicted_nll = None
    candidate_logprobs = None
    authoritative_key = None
    for sample in samples:
        mode = sample.metadata.get("noise_mode")
        nll = sample.metadata.get("nll")
        if mode == "clean" and nll is not None:
            clean_nll = float(nll)
        elif mode in {"irrelevant", "hard_distractor"} and nll is not None:
            distracted_nll = float(nll)
        elif mode == "conflict" and nll is not None:
            conflicted_nll = float(nll)
        if "candidate_logprobs" in sample.metadata:
            candidate_logprobs = {
                str(key): float(value)
                for key, value in sample.metadata["candidate_logprobs"].items()
            }
            authoritative_key = sample.metadata.get("authoritative_key")

    out = {
        "distractor_sensitivity": (
            distractor_sensitivity(clean_nll, distracted_nll)
            if clean_nll is not None and distracted_nll is not None
            else None
        ),
        "conflict_sensitivity": (
            conflict_sensitivity(clean_nll, conflicted_nll)
            if clean_nll is not None and conflicted_nll is not None
            else None
        ),
    }
    if candidate_logprobs:
        out["conflict_entropy"] = conflict_entropy(candidate_logprobs)
        if authoritative_key is not None:
            out["authoritative_margin"] = authoritative_margin(
                candidate_logprobs,
                str(authoritative_key),
            )
    return out


def _safe_section(name: str, fn) -> dict:
    try:
        return fn()
    except Exception as exc:
        return {"status": "fail", "error": f"{name} failed: {exc}"}


def _overall_status(statuses: list[str]) -> str:
    if "fail" in statuses:
        return "fail"
    if "warning" in statuses:
        return "warning"
    if "unknown" in statuses:
        return "warning"
    return "pass"


def _gpu_suite_findings(statuses: list[str]) -> list[str]:
    findings = []
    if "fail" in statuses:
        findings.append(
            "One or more GPU metric sections failed; inspect section errors before using values."
        )
    else:
        findings.append(
            "Logits-level metrics were computed with OpenVINO Runtime on the target GPU."
        )
    if "unknown" in statuses:
        findings.append("Judge-based or optional metrics are unknown without an external scorer.")
    return findings


def _compute_likelihood_section(runner, samples) -> dict:
    sample_metrics = []
    nll_sum = 0.0
    token_count = 0
    for sample in samples:
        encoded = _encode_for_runner(runner, _sample_text(sample))
        result = nll_ppl_from_logits(
            runner.forward_logits(encoded["input_ids"], encoded.get("attention_mask")),
            encoded["input_ids"],
            encoded.get("attention_mask"),
        )
        sample_metrics.append({"id": sample.id, **result})
        nll_sum += float(result["nll"]) * int(result["num_tokens"])
        token_count += int(result["num_tokens"])
    mean_nll = nll_sum / max(token_count, 1)
    return {
        "nll": mean_nll,
        "perplexity": float(np.exp(mean_nll)),
        "num_tokens": token_count,
        "samples": sample_metrics,
        "status": "pass",
    }


def _compute_self_drift_section(runner, samples) -> dict:
    aggregates = []
    sample_metrics = []
    for sample in samples:
        encoded = _encode_for_runner(runner, _sample_text(sample))
        logits = runner.forward_logits(
            encoded["input_ids"],
            encoded.get("attention_mask"),
        )[:, :-1, :]
        agg = aggregate_drift(distribution_drift(logits, logits))
        topk = {
            "top10_overlap": topk_overlap(logits, logits, k=10),
            "top1_changed_rate": top1_changed_rate(logits, logits),
        }
        sample_metrics.append({"id": sample.id, **agg, **topk})
        aggregates.append(agg)
    report_agg = {key: float(np.mean([item[key] for item in aggregates])) for key in aggregates[0]}
    report_agg["top10_overlap"] = float(np.mean([item["top10_overlap"] for item in sample_metrics]))
    report_agg["top1_changed_rate"] = float(
        np.mean([item["top1_changed_rate"] for item in sample_metrics])
    )
    return {**report_agg, "samples": sample_metrics, "status": "pass"}


def _compute_long_context_section(runner, sample, window_size: int, stride: int) -> dict:
    base_text = _sample_text(sample)
    long_text = " ".join([base_text] * max(8, window_size // max(len(base_text.split()), 1)))
    encoded = _encode_for_runner(runner, long_text)
    result = sliding_window_ppl(
        runner,
        encoded["input_ids"],
        encoded.get("attention_mask"),
        window_size=window_size,
        stride=stride,
    )
    position_buckets = aggregate_position_bucketed_ppl(
        result["per_token"],
        seq_len=int(encoded["input_ids"].shape[1]),
    )
    position_ppl = {bucket: values["ppl"] for bucket, values in position_buckets.items()}
    lim_score = (
        lost_in_middle_score_from_ppl(position_ppl)
        if {"0_10", "30_50", "50_70", "90_100"}.issubset(position_ppl)
        else None
    )
    return {
        "nll": result["nll"],
        "perplexity": result["perplexity"],
        "num_tokens": result["num_tokens"],
        "position_bucketed_ppl": position_buckets,
        "lost_in_middle_score": lim_score,
        "status": "pass" if lim_score is not None else "unknown",
    }


def _compute_serving_section(runner, samples) -> dict:
    if len(samples) < 2:
        return {"status": "unknown", "warnings": ["Need at least two samples for batch drift"]}
    first = _encode_for_runner(runner, _sample_text(samples[0]))
    second = _encode_for_runner(runner, _sample_text(samples[1]))
    alone_logits = runner.forward_logits(first["input_ids"], first.get("attention_mask"))
    batched_ids, batched_mask = _pad_batch([first["input_ids"][0], second["input_ids"][0]])
    batched_logits = runner.forward_logits(batched_ids, batched_mask)
    seq_len = first["input_ids"].shape[1]
    drift = batch_invariance_drift(
        alone_logits[:, : seq_len - 1, :],
        batched_logits[:1, : seq_len - 1, :],
    )
    drift["top10_overlap"] = topk_overlap(
        alone_logits[:, : seq_len - 1, :],
        batched_logits[:1, : seq_len - 1, :],
        k=10,
    )
    return {
        "batch_invariance": drift,
        **kv_cache_drift_interface(),
        "status": "pass",
    }


def _compute_generation_section(genai_model: str | None, device: str) -> dict:
    if genai_model is None:
        return {
            "status": "unknown",
            "warnings": ["Pass --genai-model to run OpenVINO GenAI generation metrics"],
        }
    try:
        from oviqs.runners.openvino_genai import OVGenAIRunner

        runner = OVGenAIRunner(genai_model, device=device)
        text = str(
            runner.generate(
                "Return a small JSON object with key status.",
                max_new_tokens=32,
            )
        )
    except Exception as exc:
        return {"status": "unknown", "warnings": [f"OpenVINO GenAI generation failed: {exc}"]}
    return {
        "sample_output": text,
        "ngram_repetition": ngram_repetition_rate(text, n=3),
        "json_validity": json_validity(text),
        "status": "pass",
    }


def _compute_serving_generation_section(runner, samples) -> dict:
    if not hasattr(runner, "generate"):
        return {
            "generation_prefix_divergence": None,
            "warnings": ["Serving generation prefix divergence requires a generation runner"],
        }
    first = _sample_text(samples[0])
    try:
        alone = str(runner.generate(first, max_new_tokens=16))
        repeated = str(runner.generate(first, max_new_tokens=16))
    except Exception as exc:
        return {
            "generation_prefix_divergence": None,
            "warnings": [f"Serving generation failed: {exc}"],
        }
    from oviqs.metrics.serving import generation_prefix_divergence

    return {
        "generation_prefix_divergence": generation_prefix_divergence(alone, repeated),
    }


def _compute_rag_section() -> dict:
    return {
        **evidence_coverage(
            ["OpenVINO", "Intel GPU"],
            ["OpenVINO can run model inference on Intel GPU devices."],
        ),
        **supported_claim_ratio_placeholder(),
        "status": "unknown",
    }


def _compute_eval_rag_section(samples, answer_rows: list[dict], scorer: str) -> dict:
    answer_by_id = {str(row.get("id")): row for row in answer_rows}
    sample_reports = []
    precision_values = []
    recall_values = []
    evidence_values = []
    citation_precision_values = []
    citation_recall_values = []
    faithfulness_values = []
    distractor_values = []

    for sample in samples:
        answer_row = answer_by_id.get(sample.id, {})
        answer = str(answer_row.get("answer") or sample.metadata.get("answer") or "")
        claims = answer_row.get("claims") or sample.metadata.get("claims") or []
        actual_citations = answer_row.get("citations") or sample.metadata.get("citations") or []
        expected_citations = sample.metadata.get("expected_citations") or sample.references
        relevant_indices = sample.metadata.get("relevant_context_indices") or []

        evidence = evidence_coverage(sample.expected_evidence, sample.retrieved_contexts)
        precision = context_precision(sample.expected_evidence, sample.retrieved_contexts)
        recall = context_recall(sample.expected_evidence, sample.retrieved_contexts)
        citations = citation_metrics(expected_citations, actual_citations)
        faithfulness = rule_based_faithfulness(answer, sample.retrieved_contexts, claims)
        distractors = distractor_ratio(sample.retrieved_contexts, relevant_indices)

        evidence_values.append(float(evidence["evidence_coverage"]))
        precision_values.append(float(precision["context_precision"]))
        recall_values.append(float(recall["context_recall"]))
        citation_precision_values.append(float(citations["citation_precision"]))
        citation_recall_values.append(float(citations["citation_recall"]))
        faithfulness_values.append(float(faithfulness["faithfulness"]))
        distractor_values.append(float(distractors["distractor_ratio"]))
        sample_reports.append(
            {
                "id": sample.id,
                **evidence,
                **precision,
                **recall,
                **citations,
                **faithfulness,
                **distractors,
            }
        )

    warnings = []
    judge_metrics = supported_claim_ratio_placeholder()
    if scorer != "placeholder":
        warnings.append(f"External RAG scorer {scorer!r} is not configured in this CLI run")
    warnings.extend(judge_metrics.get("warnings", []))
    return {
        "evidence_coverage": _mean(evidence_values),
        "context_precision": _mean(precision_values),
        "context_recall": _mean(recall_values),
        "citation_precision": _mean(citation_precision_values),
        "citation_recall": _mean(citation_recall_values),
        "faithfulness": _mean(faithfulness_values),
        "distractor_ratio": _mean(distractor_values),
        "supported_claim_ratio": judge_metrics["supported_claim_ratio"],
        "samples": sample_reports,
        "warnings": warnings,
        "status": "warning" if warnings else "pass",
    }


def _compute_agent_section() -> dict:
    trace = AgentTrace(
        id="gpu_suite_agent_trace",
        input="Find GPU metric report",
        steps=[
            TraceStep(type="tool_call", tool="search", args={"query": "gpu report"}),
            TraceStep(type="tool_call", tool="search", args={"query": "gpu report"}),
            TraceStep(type="final", content="Report found."),
        ],
    )
    return {
        **tool_call_validity(trace, {"search": {"required": ["query"]}}),
        **redundant_tool_call_rate(trace),
        **compute_agent_state_drift({"report_found": True}, {"report_found": True}),
        **task_completion(trace),
        **policy_violation_rate(trace),
        **recovery_after_tool_error(trace),
        **observation_grounding_score_placeholder(),
        "status": "unknown",
    }


def _compute_eval_agent_section(trace_rows: list[dict], expected_rows: list[dict]) -> dict:
    expected_by_id = {str(row.get("id")): row for row in expected_rows}
    sample_reports = []
    validity_values = []
    redundancy_values = []
    state_drift_values = []
    grounding_values = []
    completion_values = []
    policy_values = []
    recovery_values = []

    for row in trace_rows:
        trace = AgentTrace.model_validate(row)
        expected = expected_by_id.get(trace.id, {})
        tool_schemas = expected.get("tool_schemas") or trace.expected_constraints.get(
            "tool_schemas",
            {},
        )
        expected_state = expected.get("expected_state") or trace.expected_state
        actual_state = expected.get("actual_state") or trace.metadata.get("actual_state", {})
        validity = tool_call_validity(trace, tool_schemas)
        redundancy = redundant_tool_call_rate(trace)
        state = compute_agent_state_drift(expected_state, actual_state)
        grounding = compute_observation_grounding_score(trace)
        completion = task_completion(trace)
        policy = policy_violation_rate(trace)
        recovery = recovery_after_tool_error(trace)

        validity_values.append(float(validity["tool_call_validity"]))
        redundancy_values.append(float(redundancy["redundant_tool_call_rate"]))
        state_drift_values.append(float(state["state_drift_score"]))
        grounding_values.append(float(grounding["observation_grounding_score"]))
        completion_values.append(float(completion["task_completion"]))
        policy_values.append(float(policy["policy_violation_rate"]))
        if recovery["recovery_after_tool_error"] is not None:
            recovery_values.append(float(recovery["recovery_after_tool_error"]))
        sample_reports.append(
            {
                "id": trace.id,
                **validity,
                **redundancy,
                **state,
                **grounding,
                **completion,
                **policy,
                **recovery,
            }
        )

    return {
        "tool_call_validity": _mean(validity_values),
        "redundant_tool_call_rate": _mean(redundancy_values),
        "agent_state_drift": _mean(state_drift_values),
        "observation_grounding_score": _mean(grounding_values),
        "task_completion": _mean(completion_values),
        "policy_violation_rate": _mean(policy_values),
        "recovery_after_tool_error": _mean(recovery_values) if recovery_values else None,
        "samples": sample_reports,
        "status": "pass",
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _pad_batch(rows: list[np.ndarray], pad_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
    max_len = max(int(row.shape[0]) for row in rows)
    input_ids = np.full((len(rows), max_len), pad_id, dtype=np.int64)
    attention_mask = np.zeros((len(rows), max_len), dtype=np.int64)
    for idx, row in enumerate(rows):
        length = int(row.shape[0])
        input_ids[idx, :length] = row
        attention_mask[idx, :length] = 1
    return input_ids, attention_mask


__all__ = [
    "app",
    "context_gain",
    "degradation_slope",
    "distractor_sensitivity",
]
