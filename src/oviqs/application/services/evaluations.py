from __future__ import annotations

import os
from typing import Any

import numpy as np

from oviqs.application.dto.requests import (
    AgentEvaluationRequest,
    DriftEvaluationRequest,
    GpuSuiteRequest,
    LikelihoodEvaluationRequest,
    LongContextEvaluationRequest,
    RagEvaluationRequest,
    ServingEvaluationRequest,
)
from oviqs.application.services.evaluation import (
    compute_agent_section,
    compute_eval_agent_section,
    compute_eval_long_context_section,
    compute_eval_rag_section,
    compute_generation_section,
    compute_likelihood_section,
    compute_long_context_section,
    compute_performance_section,
    compute_rag_section,
    compute_reference_drift_section,
    compute_self_drift_section,
    compute_serving_generation_section,
    compute_serving_section,
    default_serving_samples,
    encode_for_runner,
    gpu_suite_findings,
    overall_status,
    safe_section,
)
from oviqs.domain.metrics.distribution_drift import aggregate_drift, distribution_drift
from oviqs.domain.metrics.likelihood import nll_ppl_from_logits, sliding_window_ppl
from oviqs.domain.reports import EvaluationReport, ReportRun, ReportSummary
from oviqs.ports.datasets import DatasetReaderPort, DatasetRowsReaderPort
from oviqs.ports.runners import GenerationRunnerFactoryPort, LogitsRunnerFactoryPort


def build_likelihood_report(
    request: LikelihoodEvaluationRequest,
    runner_factory: LogitsRunnerFactoryPort,
    dataset_reader: DatasetReaderPort,
) -> EvaluationReport:
    runner = runner_factory(request.backend, request.model, request.device)
    section = _compute_windowed_likelihood_section(
        runner,
        dataset_reader.read_samples(request.dataset),
        request.window_size,
        request.stride,
    )
    return EvaluationReport(
        run=ReportRun(
            id=request.out.stem,
            model=request.model,
            current=request.backend,
            device=request.device,
        ),
        summary=ReportSummary(overall_status="pass"),
        likelihood=section,
    )


def build_drift_report(
    request: DriftEvaluationRequest,
    runner_factory: LogitsRunnerFactoryPort,
    dataset_reader: DatasetReaderPort,
) -> EvaluationReport:
    ref_runner = runner_factory(
        request.reference_backend,
        request.reference,
        request.reference_device,
    )
    cur_runner = runner_factory(request.current_backend, request.current, request.device)
    sample_metrics = []
    aggregates = []
    for sample in dataset_reader.read_samples(request.dataset):
        text = _sample_text(sample)
        encoded = encode_for_runner(ref_runner, text)
        ref_logits = ref_runner.forward_logits(encoded["input_ids"], encoded.get("attention_mask"))[
            :, :-1, :
        ]
        cur_logits = cur_runner.forward_logits(encoded["input_ids"], encoded.get("attention_mask"))[
            :, :-1, :
        ]
        if ref_logits.shape != cur_logits.shape:
            raise ValueError(
                f"Logits shape mismatch for sample {sample.id}: "
                f"{ref_logits.shape} vs {cur_logits.shape}"
            )
        agg = aggregate_drift(distribution_drift(ref_logits, cur_logits))
        sample_metrics.append({"id": sample.id, **agg})
        aggregates.append(agg)
    report_agg = (
        {key: float(np.mean([item[key] for item in aggregates])) for key in aggregates[0]}
        if aggregates
        else {}
    )
    return EvaluationReport(
        run=ReportRun(
            id=request.out.stem,
            reference=request.reference_backend,
            current=request.current_backend,
            device=request.device,
        ),
        summary=ReportSummary(overall_status="pass"),
        inference_equivalence={**report_agg, "samples": sample_metrics},
    )


def build_long_context_report(
    request: LongContextEvaluationRequest,
    runner_factory: LogitsRunnerFactoryPort,
    dataset_reader: DatasetReaderPort,
) -> EvaluationReport:
    runner = runner_factory(request.backend, request.model, request.device)
    samples = dataset_reader.read_samples(request.dataset)
    if not samples:
        raise ValueError("long-context dataset must contain at least one sample")
    section = compute_eval_long_context_section(
        runner,
        samples,
        list(request.lengths),
        request.window_size,
        request.stride,
    )
    return EvaluationReport(
        run=ReportRun(
            id=request.out.stem,
            model=request.model,
            current=request.backend,
            device=request.device,
        ),
        summary=ReportSummary(
            overall_status=overall_status([section.get("status", "unknown")]),
            main_findings=["Long-context metrics were computed from the provided JSONL samples."],
        ),
        long_context=section,
    )


def build_serving_report(
    request: ServingEvaluationRequest,
    runner_factory: LogitsRunnerFactoryPort,
    generation_runner_factory: GenerationRunnerFactoryPort,
    dataset_reader: DatasetReaderPort,
) -> EvaluationReport:
    samples = (
        dataset_reader.read_samples(request.dataset)
        if request.dataset
        else default_serving_samples()
    )
    runner = runner_factory(request.backend, request.model, request.device)
    serving = safe_section("serving", lambda: compute_serving_section(runner, samples))
    generation_runner: Any = runner
    try:
        generation_runner = generation_runner_factory(
            request.backend,
            request.model,
            request.device,
        )
    except ValueError:
        generation_runner = runner
    generation = compute_serving_generation_section(generation_runner, samples)
    return EvaluationReport(
        run=ReportRun(
            id=request.out.stem,
            model=request.model,
            current=request.backend,
            device=request.device,
        ),
        summary=ReportSummary(overall_status=overall_status([serving.get("status", "unknown")])),
        serving={**serving, **generation},
    )


def build_rag_report(
    request: RagEvaluationRequest,
    dataset_reader: DatasetReaderPort,
    rows_reader: DatasetRowsReaderPort,
) -> EvaluationReport:
    samples = dataset_reader.read_samples(request.dataset)
    answer_rows = rows_reader.read_rows(request.answers) if request.answers else []
    rag = compute_eval_rag_section(samples, answer_rows, request.scorer)
    return EvaluationReport(
        run=ReportRun(id=request.out.stem),
        summary=ReportSummary(overall_status=overall_status([rag.get("status", "unknown")])),
        rag=rag,
    )


def build_agent_report(
    request: AgentEvaluationRequest,
    rows_reader: DatasetRowsReaderPort,
) -> EvaluationReport:
    trace_rows = rows_reader.read_rows(request.traces)
    expected_rows = rows_reader.read_rows(request.expected) if request.expected else []
    agent = compute_eval_agent_section(trace_rows, expected_rows)
    return EvaluationReport(
        run=ReportRun(id=request.out.stem),
        summary=ReportSummary(overall_status=overall_status([agent.get("status", "unknown")])),
        agent=agent,
    )


def build_gpu_suite_report(
    request: GpuSuiteRequest,
    runner_factory: LogitsRunnerFactoryPort,
    generation_runner_factory: GenerationRunnerFactoryPort,
    dataset_reader: DatasetReaderPort,
) -> EvaluationReport:
    runner = runner_factory(request.backend, request.model, request.device)
    samples = dataset_reader.read_samples(request.dataset)
    if not samples:
        raise ValueError("GPU suite dataset must contain at least one sample")

    reference_runner = None
    if os.environ.get("OVIQS_ENABLE_CPU_REFERENCE") in {"1", "true", "True", "yes"}:
        try:
            reference_runner = runner_factory(request.backend, request.model, "CPU")
        except Exception:
            reference_runner = None

    likelihood = safe_section(
        "likelihood",
        lambda: compute_likelihood_section(runner, samples, reference_runner),
    )
    inference_equivalence = safe_section(
        "inference_equivalence",
        lambda: (
            compute_reference_drift_section(reference_runner, runner, samples)
            if reference_runner is not None
            else compute_self_drift_section(runner, samples)
        ),
    )
    long_context = safe_section(
        "long_context",
        lambda: compute_long_context_section(
            runner, samples[0], request.window_size, request.stride
        ),
    )
    serving = safe_section("serving", lambda: compute_serving_section(runner, samples))
    generation = safe_section(
        "generation",
        lambda: compute_generation_section(
            request.genai_model,
            request.device,
            generation_runner_factory,
        ),
    )
    rag = safe_section("rag", compute_rag_section)
    agent = safe_section("agent", compute_agent_section)
    performance = safe_section(
        "performance", lambda: compute_performance_section(runner, samples[0])
    )
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
            performance,
        ]
    ]

    return EvaluationReport(
        run=ReportRun(
            id=request.out.stem,
            suite="openvino_llm_quality_v1_gpu",
            model=request.model,
            current=request.backend,
            device=request.device,
        ),
        summary=ReportSummary(
            overall_status=overall_status(section_statuses),
            main_findings=gpu_suite_findings(section_statuses),
        ),
        inference_equivalence=inference_equivalence,
        likelihood=likelihood,
        long_context=long_context,
        generation=generation,
        rag=rag,
        agent=agent,
        serving=serving,
        performance=performance,
    )


def _compute_windowed_likelihood_section(
    runner: Any,
    samples: list[Any],
    window_size: int,
    stride: int,
) -> dict[str, Any]:
    sample_metrics = []
    nll_sum = 0.0
    token_count = 0
    for sample in samples:
        encoded = encode_for_runner(runner, _sample_text(sample))
        result = (
            sliding_window_ppl(
                runner,
                encoded["input_ids"],
                encoded.get("attention_mask"),
                window_size,
                stride,
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
    return {
        "nll": mean_nll,
        "perplexity": float(np.exp(mean_nll)),
        "num_tokens": token_count,
        "samples": sample_metrics,
    }


def _sample_text(sample: Any) -> str:
    return sample.text or " ".join(
        part for part in [sample.context, sample.prompt, sample.target] if part
    )
