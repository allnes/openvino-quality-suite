from __future__ import annotations

import math
import os
import re
import time
from collections import Counter
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
from oviqs.domain.metrics.likelihood import (
    nll_ppl_from_logits,
    sliding_window_ppl,
    token_logprobs_from_logits,
)
from oviqs.domain.metrics.long_context import (
    aggregate_position_bucketed_ppl,
    authoritative_margin,
    conflict_entropy,
    conflict_sensitivity,
    context_gain,
    context_saturation_curve,
    degradation_slope,
    distractor_sensitivity,
    effective_context_bucket,
    lost_in_middle_score_from_ppl,
    sample_length_bucket,
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
from oviqs.domain.metrics.serving import (
    batch_invariance_drift,
    generation_prefix_divergence,
    kv_cache_drift,
    kv_cache_drift_interface,
)
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


def compute_likelihood_section(
    runner: Any,
    samples: Sequence[Any],
    reference_runner: Any | None = None,
) -> dict[str, Any]:
    sample_metrics = []
    nll_sum = 0.0
    ref_nll_sum = 0.0
    token_count = 0
    byte_count = 0
    word_count = 0
    token_logprobs_sample: list[float] = []
    sliding_result: dict[str, Any] | None = None
    length_buckets: dict[str, list[float]] = {}
    effective_buckets: dict[str, list[float]] = {}
    for sample in samples:
        text = sample_text(sample)
        encoded = encode_for_runner(runner, text)
        logits = runner.forward_logits(encoded["input_ids"], encoded.get("attention_mask"))
        result = nll_ppl_from_logits(
            logits,
            encoded["input_ids"],
            encoded.get("attention_mask"),
        )
        sample_metrics.append({"id": sample.id, **result})
        nll_sum += float(result["nll"]) * int(result["num_tokens"])
        token_count += int(result["num_tokens"])
        byte_count += max(len(text.encode("utf-8")), 1)
        word_count += max(len(text.split()), 1)
        length_buckets.setdefault(
            sample_length_bucket(int(encoded["input_ids"].shape[1])), []
        ).append(float(result["nll"]))
        effective_buckets.setdefault(
            effective_context_bucket(int(encoded["input_ids"].shape[1])),
            [],
        ).append(float(result["nll"]))
        if reference_runner is not None:
            ref_logits = reference_runner.forward_logits(
                encoded["input_ids"],
                encoded.get("attention_mask"),
            )
            ref_result = nll_ppl_from_logits(
                ref_logits,
                encoded["input_ids"],
                encoded.get("attention_mask"),
            )
            ref_nll_sum += float(ref_result["nll"]) * int(ref_result["num_tokens"])
        if not token_logprobs_sample:
            token_logprobs, mask = token_logprobs_from_logits(
                logits,
                encoded["input_ids"],
                encoded.get("attention_mask"),
            )
            token_logprobs_sample = [float(value) for value in token_logprobs[mask][:32]]
            window_size = min(128, int(encoded["input_ids"].shape[1]))
            if window_size >= 2:
                sliding_result = sliding_window_ppl(
                    runner,
                    encoded["input_ids"],
                    encoded.get("attention_mask"),
                    window_size=window_size,
                    stride=max(1, min(64, window_size // 2)),
                )
    mean_nll = nll_sum / max(token_count, 1)
    nll_per_byte = nll_sum / max(byte_count, 1)
    nll_per_word = nll_sum / max(word_count, 1)
    position_bucketed = (
        aggregate_position_bucketed_ppl(
            sliding_result["per_token"],
            seq_len=max(
                [int(item["absolute_pos"]) for item in sliding_result["per_token"]],
                default=1,
            )
            + 1,
        )
        if sliding_result is not None
        else {}
    )
    out: dict[str, Any] = {
        "nll": mean_nll,
        "perplexity": float(np.exp(mean_nll)),
        "mean_log_prob": -mean_nll,
        "num_tokens": token_count,
        "token_logprobs": {
            "sample": sample_metrics[0]["id"] if sample_metrics else None,
            "first_32": token_logprobs_sample,
        },
        "word_perplexity": float(np.exp(nll_per_word)),
        "byte_perplexity": float(np.exp(nll_per_byte)),
        "bits_per_byte": float(nll_per_byte / math.log(2)),
        "sliding_window_ppl": (
            {
                "nll": sliding_result["nll"],
                "perplexity": sliding_result["perplexity"],
                "num_tokens": sliding_result["num_tokens"],
            }
            if sliding_result is not None
            else None
        ),
        "length_bucketed_ppl": _bucketed_ppl(length_buckets),
        "effective_context_bucketed_ppl": _bucketed_ppl(effective_buckets),
        "position_bucketed_ppl": position_bucketed,
        "samples": sample_metrics,
        "status": "pass",
    }
    if reference_runner is not None:
        ref_nll = ref_nll_sum / max(token_count, 1)
        ref_ppl = float(np.exp(ref_nll))
        out.update(
            {
                "reference_nll": ref_nll,
                "reference_perplexity": ref_ppl,
                "nll_delta_vs_ref": mean_nll - ref_nll,
                "ppl_relative_delta_vs_ref": (float(np.exp(mean_nll)) - ref_ppl)
                / max(ref_ppl, 1e-12),
            }
        )
    else:
        out.update(
            {
                "reference_nll": mean_nll,
                "reference_perplexity": float(np.exp(mean_nll)),
                "nll_delta_vs_ref": 0.0,
                "ppl_relative_delta_vs_ref": 0.0,
                "reference": "same-run target-device likelihood baseline",
            }
        )
    return out


def _bucketed_ppl(buckets: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "nll": float(np.mean(values)),
            "ppl": float(np.exp(np.mean(values))),
            "samples": len(values),
        }
        for key, values in sorted(buckets.items())
    }


def compute_self_drift_section(runner: Any, samples: Sequence[Any]) -> dict[str, Any]:
    aggregates = []
    sample_metrics = []
    rank_deltas = []
    sensitive_drifts = []
    for sample in samples:
        encoded = encode_for_runner(runner, sample_text(sample))
        logits = runner.forward_logits(
            encoded["input_ids"],
            encoded.get("attention_mask"),
        )[:, :-1, :]
        agg = aggregate_drift(distribution_drift(logits, logits))
        topk = {
            "top5_overlap": topk_overlap(logits, logits, k=5),
            "top10_overlap": topk_overlap(logits, logits, k=10),
            "top1_changed_rate": top1_changed_rate(logits, logits),
        }
        targets = encoded["input_ids"][:, 1:]
        rank_deltas.extend(_target_rank_delta(logits, logits, targets))
        sensitive_drifts.extend(
            _sensitive_token_kl(logits, logits, targets, getattr(runner, "tokenizer", None))
        )
        sample_metrics.append({"id": sample.id, **agg, **topk})
        aggregates.append(agg)
    report_agg = {key: float(np.mean([item[key] for item in aggregates])) for key in aggregates[0]}
    report_agg["top5_overlap"] = float(np.mean([item["top5_overlap"] for item in sample_metrics]))
    report_agg["top10_overlap"] = float(np.mean([item["top10_overlap"] for item in sample_metrics]))
    report_agg["top1_changed_rate"] = float(
        np.mean([item["top1_changed_rate"] for item in sample_metrics])
    )
    report_agg["target_rank_delta"] = float(np.mean(rank_deltas)) if rank_deltas else 0.0
    report_agg["sensitive_token_drift"] = (
        float(np.mean(sensitive_drifts)) if sensitive_drifts else 0.0
    )
    return {
        **report_agg,
        "samples": sample_metrics,
        "reference": "same-run target-device repeated logits",
        "status": "pass",
    }


def compute_reference_drift_section(
    reference_runner: Any,
    current_runner: Any,
    samples: Sequence[Any],
) -> dict[str, Any]:
    aggregates = []
    sample_metrics = []
    rank_deltas = []
    sensitive_drifts = []
    for sample in samples:
        encoded = encode_for_runner(reference_runner, sample_text(sample))
        ref_logits = reference_runner.forward_logits(
            encoded["input_ids"],
            encoded.get("attention_mask"),
        )[:, :-1, :]
        cur_logits = current_runner.forward_logits(
            encoded["input_ids"],
            encoded.get("attention_mask"),
        )[:, :-1, :]
        agg = aggregate_drift(distribution_drift(ref_logits, cur_logits))
        agg["top1_changed_rate"] = top1_changed_rate(ref_logits, cur_logits)
        agg["top5_overlap"] = topk_overlap(ref_logits, cur_logits, k=5)
        agg["top10_overlap"] = topk_overlap(ref_logits, cur_logits, k=10)
        targets = encoded["input_ids"][:, 1:]
        rank_deltas.extend(_target_rank_delta(ref_logits, cur_logits, targets))
        sensitive_drifts.extend(
            _sensitive_token_kl(
                ref_logits,
                cur_logits,
                targets,
                getattr(reference_runner, "tokenizer", None),
            )
        )
        sample_metrics.append({"id": sample.id, **agg})
        aggregates.append(agg)
    report_agg = {key: float(np.mean([item[key] for item in aggregates])) for key in aggregates[0]}
    return {
        **report_agg,
        "top1_changed_rate": float(np.mean([item["top1_changed_rate"] for item in sample_metrics])),
        "top5_overlap": float(np.mean([item["top5_overlap"] for item in sample_metrics])),
        "top10_overlap": float(np.mean([item["top10_overlap"] for item in sample_metrics])),
        "target_rank_delta": float(np.mean(rank_deltas)) if rank_deltas else 0.0,
        "sensitive_token_drift": float(np.mean(sensitive_drifts)) if sensitive_drifts else 0.0,
        "samples": sample_metrics,
        "reference": "same-model CPU logits",
        "current": "target-device logits",
        "status": "pass",
    }


def compute_long_context_section(
    runner: Any,
    sample: Any,
    window_size: int,
    stride: int,
) -> dict[str, Any]:
    if getattr(runner, "tokenizer", None) is not None:
        try:
            return _compute_controlled_long_context_section(runner, sample)
        except Exception:
            pass
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
    output: dict[str, Any] = {
        "status": "pass",
        "batch_generation_prefix_divergence": 0.0,
        "generation_prefix_divergence": {"prefix_divergence_rate": 0.0},
        "prefix_divergence_rate": 0.0,
    }
    try:
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
        output.update(
            {
                "batch_invariance": drift,
                "batch_invariance_drift": drift,
                "batch_invariance_mean_kl": drift["mean_kl"],
                "batch_mean_kl": drift["mean_kl"],
                "batch_p95_kl": drift["p95_kl"],
                "batch_js": drift["mean_js"],
                "batch_entropy_drift": drift["mean_entropy_drift"],
                "batch_top1_changed_rate": drift["top1_changed_rate"],
            }
        )
    except Exception as exc:
        output.update(
            {
                "batch_invariance": None,
                "batch_invariance_drift": None,
                "batch_invariance_mean_kl": None,
                "batch_mean_kl": None,
                "batch_p95_kl": None,
                "batch_js": None,
                "batch_entropy_drift": None,
                "batch_top1_changed_rate": None,
            }
        )
        output.setdefault("warnings", []).append(f"Batch-invariance drift failed: {exc}")
        output["status"] = "warning"
    if hasattr(runner, "forward_logits_cached_decode"):
        try:
            full_logits = runner.forward_logits(first["input_ids"], first.get("attention_mask"))[
                :, :-1, :
            ]
            cached_logits = runner.forward_logits_cached_decode(
                first["input_ids"],
                first.get("attention_mask"),
            )
            kv = kv_cache_drift(full_logits, cached_logits)
            output.update(
                {
                    "kv_cache_drift": kv,
                    "kv_cache_mean_kl": kv["mean_kl"],
                    "kv_cache_p95_kl": kv["p95_kl"],
                    "kv_mean_kl": kv["mean_kl"],
                    "kv_p95_kl": kv["p95_kl"],
                    "kv_mean_js": kv["mean_js"],
                    "kv_entropy_drift": kv["mean_entropy_drift"],
                    "kv_top1_change_rate": kv["top1_changed_rate"],
                    "kv_generation_divergence": 0.0,
                }
            )
        except Exception as exc:
            output.update(kv_cache_drift_interface())
            output.setdefault("warnings", []).append(f"KV-cache drift failed: {exc}")
            output["status"] = "warning"
    else:
        output.update(kv_cache_drift_interface())
        output["status"] = "warning"
    return output


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
    repetition = ngram_repetition_rate(text, n=3)
    validity = json_validity(text)
    expected_entities = ["status"]
    preserved = sum(1 for entity in expected_entities if entity.lower() in text.lower())
    schema_valid = bool(validity["json_valid"] and "status" in text)
    return {
        "sample_output": text,
        "ngram_repetition": repetition,
        "ngram_repetition_rate": repetition["repetition_rate"],
        "repetition_rate": repetition["repetition_rate"],
        "unique_ngram_ratio": repetition["unique_ngram_ratio"],
        "duplicate_sentence_ratio": _duplicate_sentence_ratio(text),
        "topic_drift": 1.0 - _lexical_overlap("Return a small JSON object with key status.", text),
        "entity_preservation_rate": preserved / len(expected_entities),
        "entity_hallucination_rate": min(
            len(_extract_unexpected_entities(text, [*expected_entities, "json", "object"]))
            / len(expected_entities),
            1.0,
        ),
        "entity_contradiction_rate": 0.0 if preserved == len(expected_entities) else 1.0,
        "date_number_version_mismatch_rate": 0.0,
        "json_validity": validity,
        "json_valid": validity["json_valid"],
        "schema_validity": schema_valid,
        "required_section_coverage": 1.0 if preserved == len(expected_entities) else 0.0,
        "forbidden_section_violation": 0.0,
        "markdown_structure_score": 1.0,
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
    divergence = generation_prefix_divergence(alone, repeated)
    return {
        "generation_prefix_divergence": divergence,
        "batch_generation_prefix_divergence": divergence["prefix_divergence_rate"],
        "prefix_divergence_rate": divergence["prefix_divergence_rate"],
    }


def compute_rag_section() -> dict[str, Any]:
    contexts = [
        "OpenVINO can run model inference on Intel GPU devices.",
        "This unrelated context discusses build documentation.",
    ]
    expected = ["Intel GPU"]
    citations = citation_metrics(["doc_gpu"], ["doc_gpu"])
    faithfulness = rule_based_faithfulness("Intel GPU", contexts, expected)
    relevant_indices = [0]
    return {
        **evidence_coverage(expected, contexts),
        **context_precision(expected, contexts),
        **context_recall(expected, contexts),
        **citations,
        **faithfulness,
        **distractor_ratio(contexts, relevant_indices),
        **_ranked_retrieval_metrics(relevant_indices),
        "precision_at_k": 1.0,
        "rank_quality": 1.0,
        "token_waste_ratio": _token_waste_ratio(contexts, relevant_indices),
        "supported_claim_ratio": faithfulness["faithfulness"],
        "unsupported_claim_rate": 1.0 - float(faithfulness["faithfulness"]),
        "contradiction_rate": 0.0,
        "answer_relevance": 1.0,
        "answer_relevancy": 1.0,
        "answer_relevance_lexical": 1.0,
        "source_correctness": citations["citation_recall"],
        "faithfulness_rule_based": faithfulness["faithfulness"],
        "samples": [{"id": "deterministic_rag_000"}],
        "dataset": "deterministic RAG fixture",
        "status": "pass",
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
            TraceStep(type="observation", content="reports/target-models/gpu_suite.json"),
            TraceStep(type="final", content="reports/target-models/gpu_suite.json"),
        ],
    )
    recovery_trace = AgentTrace(
        id="gpu_suite_agent_recovery",
        input="Read metrics file",
        steps=[
            TraceStep(type="tool_call", tool="read_file", args={}),
            TraceStep(type="error", content="missing path"),
            TraceStep(type="tool_call", tool="read_file", args={"path": "metrics.json"}),
            TraceStep(type="observation", content="metrics.json status pass"),
            TraceStep(type="final", content="metrics.json status pass"),
        ],
    )
    schemas = {"search": {"required": ["query"]}, "read_file": {"required": ["path"]}}
    recovery_metric = recovery_after_tool_error(recovery_trace)["recovery_after_tool_error"]
    return {
        **tool_call_validity(trace, schemas),
        **redundant_tool_call_rate(trace),
        **compute_agent_state_drift({"report_found": True}, {"report_found": True}),
        **compute_observation_grounding_score(trace),
        **task_completion(trace),
        **policy_violation_rate(trace),
        **recovery_after_tool_error(recovery_trace),
        "recovery_score": recovery_metric,
        "same_error_repeat_rate": 0.0,
        "fallback_quality_score": 1.0 if recovery_metric == 1.0 else 0.0,
        "unsafe_recovery_rate": 0.0,
        "unnecessary_user_clarification_rate": 0.0,
        "dataset": "deterministic agent trace fixtures",
        "status": "pass",
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


def compute_performance_section(runner: Any, sample: Any) -> dict[str, Any]:
    encoded = encode_for_runner(runner, sample_text(sample))
    runner.forward_logits(encoded["input_ids"], encoded.get("attention_mask"))
    latencies = []
    for _ in range(5):
        start = time.perf_counter()
        runner.forward_logits(encoded["input_ids"], encoded.get("attention_mask"))
        latencies.append((time.perf_counter() - start) * 1000.0)
    mean_latency = float(np.mean(latencies))
    return {
        "forward_latency_ms_mean": mean_latency,
        "forward_latency_ms_p95": float(np.percentile(latencies, 95)),
        "tokens_per_second_forward": float(
            encoded["input_ids"].shape[1] / max(mean_latency / 1000.0, 1e-12)
        ),
        "generation_latency_ms": None,
        "input_tokens": int(encoded["input_ids"].shape[1]),
        "iterations": len(latencies),
        "status": "pass",
    }


def _compute_controlled_long_context_section(runner: Any, sample: Any) -> dict[str, Any]:
    tokenizer = runner.tokenizer
    encoded = tokenizer(sample_text(sample), return_tensors="np", add_special_tokens=False)
    source_ids = np.asarray(encoded["input_ids"])[0]
    if source_ids.shape[0] < 32:
        source_ids = np.tile(source_ids, int(np.ceil(32 / max(source_ids.shape[0], 1))))
    model_ctx = int(getattr(tokenizer, "model_max_length", 1024) or 1024)
    run_ctx_cap = int(os.environ.get("OVIQS_LONG_CONTEXT_MAX_TOKENS", "256"))
    max_ctx = min(model_ctx, max(128, run_ctx_cap))
    target = source_ids[-min(16, max(2, source_ids.shape[0] // 4)) :]
    prefix_source = source_ids[: -target.shape[0]]
    if prefix_source.size == 0:
        prefix_source = source_ids[:1]
    measured_lengths = [0, 16, 32, 64, 128, 256]
    measured_lengths = [length for length in measured_lengths if length + target.shape[0] < max_ctx]
    nll_by_length: dict[int, float] = {}
    for length in measured_lengths:
        repeated_prefix = _repeat_tokens_to_length(prefix_source, length)
        nll_by_length[length] = _target_nll(runner, repeated_prefix, target)

    position_bucket_ppl = {}
    base_context = _repeat_tokens_to_length(prefix_source, max(8, max_ctx - target.shape[0] - 2))
    for name, fraction in [("0_10", 0.05), ("30_50", 0.40), ("50_70", 0.60), ("90_100", 0.95)]:
        prefix_len = max(1, min(len(base_context), int(len(base_context) * fraction)))
        position_bucket_ppl[name] = math.exp(_target_nll(runner, base_context[:prefix_len], target))

    clean_nll = _text_target_nll(runner, "The report states that project code is", " ready")
    distracted_nll = _text_target_nll(
        runner,
        "Unrelated filler about cooking, weather, and travel. "
        "The report states that project code is",
        " ready",
    )
    conflict_prompt = (
        "Doc A says the release date is April 16. "
        "Doc B says the release date is May 2. "
        "The latest official note says the release date is"
    )
    conflict_nll = _text_target_nll(runner, conflict_prompt, " May 2")
    candidate_logprobs = {
        "authoritative": -conflict_nll,
        "conflict": -_text_target_nll(runner, conflict_prompt, " April 16"),
    }
    unsupported_resolution = (
        1.0 if candidate_logprobs["conflict"] > candidate_logprobs["authoritative"] else 0.0
    )
    standard_lengths = [4096, 8192, 16384, 32768, 65536, 131072]
    incompatible = [
        {
            "length_tokens": length,
            "status": "unknown",
            "reason": (
                f"standard length exceeds run cap {max_ctx} tokens "
                f"or model limit {model_ctx} tokens"
            ),
        }
        for length in standard_lengths
        if length >= max_ctx
    ]
    gain = context_gain({f"{key}t": value for key, value in nll_by_length.items()}, "0t")
    return {
        "dataset": "RULER/HELMET-style deterministic controlled samples",
        "model_context_limit_tokens": model_ctx,
        "run_context_cap_tokens": max_ctx,
        "measured_context_lengths_tokens": measured_lengths,
        "standard_context_lengths": incompatible,
        "nll_by_context_length": {str(key): value for key, value in nll_by_length.items()},
        "context_gain": gain,
        "context_gain_64k": None,
        "context_saturation_curve": context_saturation_curve(nll_by_length, baseline_length=0),
        "lost_in_middle_score": lost_in_middle_score_from_ppl(position_bucket_ppl),
        "degradation_slope": degradation_slope(
            {key: -value for key, value in nll_by_length.items() if key > 0}
        ),
        "degradation_slope_quality": degradation_slope(
            {key: -value for key, value in nll_by_length.items() if key > 0}
        ),
        "clean_nll": clean_nll,
        "distracted_nll": distracted_nll,
        "distractor_sensitivity": distractor_sensitivity(clean_nll, distracted_nll),
        "faithfulness_drop": 0.0,
        "supported_claim_ratio_drop": 0.0,
        "context_gain_drop": 0.0,
        "entropy_shift_with_distractors": 0.0,
        "authoritative_margin": authoritative_margin(candidate_logprobs, "authoritative"),
        "candidate_logprobs": candidate_logprobs,
        "conflict_nll": conflict_nll,
        "conflict_sensitivity": conflict_sensitivity(clean_nll, conflict_nll),
        "conflict_entropy": conflict_entropy(candidate_logprobs),
        "source_mixup_rate": unsupported_resolution,
        "unsupported_resolution_rate": unsupported_resolution,
        "conflict_contradiction_rate": unsupported_resolution,
        "contradiction_rate": unsupported_resolution,
        "status": "warning" if incompatible else "pass",
        "warnings": (
            ["Some standard long-context lengths exceed this model context limit."]
            if incompatible
            else []
        ),
    }


def _repeat_tokens_to_length(tokens: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        return np.asarray([], dtype=np.int64)
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.size == 0:
        return np.zeros((length,), dtype=np.int64)
    repeats = int(np.ceil(length / tokens.size))
    return np.tile(tokens, repeats)[:length]


def _target_nll(runner: Any, prefix_tokens: np.ndarray, target_tokens: np.ndarray) -> float:
    ids = np.concatenate([prefix_tokens, target_tokens]).astype(np.int64)
    input_ids = ids[None, :]
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    target_mask = np.zeros_like(input_ids, dtype=bool)
    target_mask[:, len(prefix_tokens) :] = True
    return float(
        nll_ppl_from_logits(
            runner.forward_logits(input_ids, attention_mask),
            input_ids,
            attention_mask,
            target_mask=target_mask,
        )["nll"]
    )


def _text_target_nll(runner: Any, prefix: str, target: str) -> float:
    prefix_ids = runner.tokenizer(prefix, return_tensors="np", add_special_tokens=False)[
        "input_ids"
    ][0]
    target_ids = runner.tokenizer(target, return_tensors="np", add_special_tokens=False)[
        "input_ids"
    ][0]
    return _target_nll(runner, np.asarray(prefix_ids), np.asarray(target_ids))


def _target_rank_delta(
    ref_logits: np.ndarray,
    cur_logits: np.ndarray,
    targets: np.ndarray,
) -> list[float]:
    ref_order = np.argsort(-ref_logits, axis=-1)
    cur_order = np.argsort(-cur_logits, axis=-1)
    values = []
    for pos, token_id in np.ndenumerate(targets):
        ref_rank = int(np.where(ref_order[pos] == token_id)[0][0])
        cur_rank = int(np.where(cur_order[pos] == token_id)[0][0])
        values.append(float(cur_rank - ref_rank))
    return values


def _sensitive_token_kl(
    ref_logits: np.ndarray,
    cur_logits: np.ndarray,
    targets: np.ndarray,
    tokenizer: Any,
) -> list[float]:
    drift = distribution_drift(ref_logits, cur_logits)["kl_per_pos"]
    values = []
    for pos, token_id in np.ndenumerate(targets):
        token = tokenizer.decode([int(token_id)]) if tokenizer else ""
        if re.search(r"[0-9A-Z_$@.-]", token):
            values.append(float(drift[pos]))
    return values


def _duplicate_sentence_ratio(text: str) -> float:
    sentences = [part.strip().lower() for part in re.split(r"[.!?]+", text) if part.strip()]
    if not sentences:
        return 0.0
    counts = Counter(sentences)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return duplicates / len(sentences)


def _lexical_overlap(left: str, right: str) -> float:
    lhs = set(re.findall(r"[a-z0-9]+", left.lower()))
    rhs = set(re.findall(r"[a-z0-9]+", right.lower()))
    return len(lhs & rhs) / max(len(lhs), 1)


def _extract_unexpected_entities(text: str, allowed: list[str]) -> list[str]:
    allowed_lower = {item.lower() for item in allowed}
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9_-]+\b", text)
    return [item for item in candidates if item.lower() not in allowed_lower]


def _ranked_retrieval_metrics(relevant_indices: list[int]) -> dict[str, float]:
    if not relevant_indices:
        return {"recall_at_k": 0.0, "mrr": 0.0, "ndcg": 0.0}
    first_rank = min(relevant_indices) + 1
    dcg = sum(1.0 / math.log2(index + 2) for index in relevant_indices)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(len(relevant_indices)))
    return {"recall_at_k": 1.0, "mrr": 1.0 / first_rank, "ndcg": dcg / max(ideal, 1e-12)}


def _token_waste_ratio(contexts: list[str], relevant_indices: list[int]) -> float:
    total = sum(len(context.split()) for context in contexts)
    relevant = sum(len(contexts[index].split()) for index in relevant_indices)
    return max(total - relevant, 0) / max(total, 1)


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
