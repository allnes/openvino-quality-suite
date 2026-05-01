from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from oviqs.domain.metrics.agent import (
    agent_state_drift as compute_agent_state_drift,
)
from oviqs.domain.metrics.agent import (
    observation_grounding_score as compute_observation_grounding_score,
)
from oviqs.domain.metrics.agent import (
    observation_grounding_score_placeholder,
    policy_violation_rate,
    recovery_after_tool_error,
    redundant_tool_call_rate,
    task_completion,
    tool_call_validity,
)
from oviqs.domain.metrics.distribution_drift import (
    aggregate_drift,
    distribution_drift,
    top1_changed_rate,
    topk_overlap,
)
from oviqs.domain.metrics.generation import json_validity, ngram_repetition_rate
from oviqs.domain.metrics.likelihood import nll_ppl_from_logits, sliding_window_ppl
from oviqs.domain.metrics.long_context import (
    aggregate_position_bucketed_ppl,
    authoritative_margin,
    conflict_entropy,
    conflict_sensitivity,
    context_gain,
    degradation_slope,
    distractor_sensitivity,
    lost_in_middle_score_from_ppl,
)
from oviqs.domain.metrics.rag import (
    citation_metrics,
    context_precision,
    context_recall,
    distractor_ratio,
    evidence_coverage,
    rule_based_faithfulness,
    supported_claim_ratio_placeholder,
)
from oviqs.domain.metrics.serving import batch_invariance_drift, kv_cache_drift_interface
from oviqs.domain.reports import Status
from oviqs.domain.samples import EvalSample
from oviqs.domain.traces import AgentTrace, TraceStep
from oviqs.ports.runners import GenerationRunnerFactoryPort


def encode_for_runner(runner: Any, text: str) -> dict[str, Any]:
    if hasattr(runner, "encode"):
        encoded = runner.encode(text)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask") if hasattr(encoded, "get") else None
        if hasattr(input_ids, "detach"):
            input_ids = input_ids.detach().cpu().numpy()
        if attention_mask is not None and hasattr(attention_mask, "detach"):
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


def sample_text(sample: Any) -> str:
    return sample.text or " ".join(
        part for part in [sample.context, sample.prompt, sample.target] if part
    )


def default_serving_samples() -> list[EvalSample]:
    return [
        EvalSample(id="serving_1", task_type="serving", text="openvino serving metrics"),
        EvalSample(id="serving_2", task_type="serving", text="batch invariance metrics"),
    ]


def compute_eval_long_context_section(
    runner: Any,
    samples: Sequence[Any],
    lengths: list[int],
    window_size: int,
    stride: int,
) -> dict[str, Any]:
    sample_reports = []
    position_bucket_inputs = []
    length_to_quality: dict[int, float] = {}
    nll_by_context: dict[str, float] = {}

    for sample in samples:
        text = sample_text(sample)
        encoded = encode_for_runner(runner, text)
        base_result = nll_ppl_from_logits(
            runner.forward_logits(encoded["input_ids"], encoded.get("attention_mask")),
            encoded["input_ids"],
            encoded.get("attention_mask"),
        )
        nll_by_context.setdefault("0k", float(base_result["nll"]))
        sample_report = {"id": sample.id, "baseline": base_result}

        sample_lengths = {}
        for length in lengths:
            long_text = expand_text_to_token_length(runner, text, length)
            long_encoded = encode_for_runner(runner, long_text)
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

    robustness = compute_long_context_robustness(samples)
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


def expand_text_to_token_length(runner: Any, text: str, target_length: int) -> str:
    if not text.strip():
        raise ValueError("Cannot expand empty text for long-context metrics")
    parts = [text]
    while True:
        expanded = " ".join(parts)
        encoded = encode_for_runner(runner, expanded)
        if encoded["input_ids"].shape[1] >= target_length:
            return expanded
        parts.append(text)


def compute_long_context_robustness(samples: Sequence[Any]) -> dict[str, Any]:
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


def safe_section(name: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:
        return {"status": "fail", "error": f"{name} failed: {exc}"}


def overall_status(statuses: Sequence[str]) -> Status:
    if "fail" in statuses:
        return "fail"
    if "warning" in statuses:
        return "warning"
    if "unknown" in statuses:
        return "warning"
    return "pass"


def gpu_suite_findings(statuses: Sequence[str]) -> list[str]:
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


def compute_likelihood_section(runner: Any, samples: Sequence[Any]) -> dict[str, Any]:
    sample_metrics = []
    nll_sum = 0.0
    token_count = 0
    for sample in samples:
        encoded = encode_for_runner(runner, sample_text(sample))
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


def compute_self_drift_section(runner: Any, samples: Sequence[Any]) -> dict[str, Any]:
    aggregates = []
    sample_metrics = []
    for sample in samples:
        encoded = encode_for_runner(runner, sample_text(sample))
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


def compute_long_context_section(
    runner: Any,
    sample: Any,
    window_size: int,
    stride: int,
) -> dict[str, Any]:
    base_text = sample_text(sample)
    long_text = " ".join([base_text] * max(8, window_size // max(len(base_text.split()), 1)))
    encoded = encode_for_runner(runner, long_text)
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


def compute_serving_section(runner: Any, samples: Sequence[Any]) -> dict[str, Any]:
    if len(samples) < 2:
        return {"status": "unknown", "warnings": ["Need at least two samples for batch drift"]}
    first = encode_for_runner(runner, sample_text(samples[0]))
    second = encode_for_runner(runner, sample_text(samples[1]))
    alone_logits = runner.forward_logits(first["input_ids"], first.get("attention_mask"))
    batched_ids, batched_mask = pad_batch([first["input_ids"][0], second["input_ids"][0]])
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


def compute_generation_section(
    genai_model: str | None,
    device: str,
    generation_runner_factory: GenerationRunnerFactoryPort,
) -> dict[str, Any]:
    if genai_model is None:
        return {
            "status": "unknown",
            "warnings": ["Pass --genai-model to run OpenVINO GenAI generation metrics"],
        }
    try:
        runner = generation_runner_factory("openvino-genai", genai_model, device)
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


def compute_serving_generation_section(runner: Any, samples: Sequence[Any]) -> dict[str, Any]:
    if not hasattr(runner, "generate"):
        return {
            "generation_prefix_divergence": None,
            "warnings": ["Serving generation prefix divergence requires a generation runner"],
        }
    first = sample_text(samples[0])
    try:
        alone = str(runner.generate(first, max_new_tokens=16))
        repeated = str(runner.generate(first, max_new_tokens=16))
    except Exception as exc:
        return {
            "generation_prefix_divergence": None,
            "warnings": [f"Serving generation failed: {exc}"],
        }
    from oviqs.domain.metrics.serving import generation_prefix_divergence

    return {
        "generation_prefix_divergence": generation_prefix_divergence(alone, repeated),
    }


def compute_rag_section() -> dict[str, Any]:
    return {
        **evidence_coverage(
            ["OpenVINO", "Intel GPU"],
            ["OpenVINO can run model inference on Intel GPU devices."],
        ),
        **supported_claim_ratio_placeholder(),
        "status": "unknown",
    }


def compute_eval_rag_section(
    samples: Sequence[Any],
    answer_rows: list[dict[str, Any]],
    scorer: str,
) -> dict[str, Any]:
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
    warnings.extend(judge_metrics.get("warnings") or [])
    return {
        "evidence_coverage": mean(evidence_values),
        "context_precision": mean(precision_values),
        "context_recall": mean(recall_values),
        "citation_precision": mean(citation_precision_values),
        "citation_recall": mean(citation_recall_values),
        "faithfulness": mean(faithfulness_values),
        "distractor_ratio": mean(distractor_values),
        "supported_claim_ratio": judge_metrics["supported_claim_ratio"],
        "samples": sample_reports,
        "warnings": warnings,
        "status": "warning" if warnings else "pass",
    }


def compute_agent_section() -> dict[str, Any]:
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


def compute_eval_agent_section(
    trace_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
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
        "tool_call_validity": mean(validity_values),
        "redundant_tool_call_rate": mean(redundancy_values),
        "agent_state_drift": mean(state_drift_values),
        "observation_grounding_score": mean(grounding_values),
        "task_completion": mean(completion_values),
        "policy_violation_rate": mean(policy_values),
        "recovery_after_tool_error": mean(recovery_values) if recovery_values else None,
        "samples": sample_reports,
        "status": "pass",
    }


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def pad_batch(rows: list[np.ndarray], pad_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
    max_len = max(int(row.shape[0]) for row in rows)
    input_ids = np.full((len(rows), max_len), pad_id, dtype=np.int64)
    attention_mask = np.zeros((len(rows), max_len), dtype=np.int64)
    for idx, row in enumerate(rows):
        length = int(row.shape[0])
        input_ids[idx, :length] = row
        attention_mask[idx, :length] = 1
    return input_ids, attention_mask
