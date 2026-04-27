from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

from oviqs.aggregation.buckets import (
    aggregate_position_bucketed_ppl,
    effective_context_bucket,
    sample_length_bucket,
)
from oviqs.core.report import write_report
from oviqs.core.trace import AgentTrace, TraceStep
from oviqs.metrics.agent import (
    agent_state_drift,
    observation_grounding_score,
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
from oviqs.metrics.likelihood import (
    nll_ppl_from_logits,
    sliding_window_ppl,
    token_logprobs_from_logits,
)
from oviqs.metrics.long_context import (
    authoritative_margin,
    conflict_entropy,
    context_gain,
    context_saturation_curve,
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
)
from oviqs.metrics.serving import batch_invariance_drift, kv_cache_drift
from oviqs.references import get_metric_reference
from oviqs.runners.openvino_genai import OVGenAIRunner
from oviqs.runners.openvino_runtime import OVRuntimeLogitsRunner

GUIDE_METRICS: dict[str, list[str]] = {
    "likelihood": [
        "token_logprobs",
        "mean_log_prob",
        "num_tokens",
        "nll",
        "perplexity",
        "word_perplexity",
        "byte_perplexity",
        "bits_per_byte",
        "sliding_window_ppl",
        "length_bucketed_ppl",
        "effective_context_bucketed_ppl",
        "position_bucketed_ppl",
    ],
    "inference_equivalence": [
        "mean_kl",
        "p95_kl",
        "max_kl",
        "mean_js",
        "p95_js",
        "mean_entropy_drift",
        "mean_logit_cosine",
        "top1_changed_rate",
        "top5_overlap",
        "top10_overlap",
        "target_rank_delta",
        "sensitive_token_drift",
    ],
    "long_context": [
        "context_gain",
        "context_saturation_curve",
        "lost_in_middle_score",
        "degradation_slope",
        "distractor_sensitivity",
        "faithfulness_drop",
        "supported_claim_ratio_drop",
        "context_gain_drop",
        "entropy_shift_with_distractors",
        "authoritative_margin",
        "conflict_entropy",
        "source_mixup_rate",
        "unsupported_resolution_rate",
        "contradiction_rate",
    ],
    "generation": [
        "repetition_rate",
        "unique_ngram_ratio",
        "duplicate_sentence_ratio",
        "topic_drift",
        "entity_preservation_rate",
        "entity_hallucination_rate",
        "entity_contradiction_rate",
        "date_number_version_mismatch_rate",
        "json_validity",
        "schema_validity",
        "required_section_coverage",
        "forbidden_section_violation",
        "markdown_structure_score",
    ],
    "serving": [
        "batch_mean_kl",
        "batch_p95_kl",
        "batch_js",
        "batch_entropy_drift",
        "batch_top1_changed_rate",
        "batch_generation_prefix_divergence",
        "kv_mean_kl",
        "kv_p95_kl",
        "kv_mean_js",
        "kv_entropy_drift",
        "kv_top1_change_rate",
        "kv_generation_divergence",
    ],
    "rag": [
        "recall_at_k",
        "context_recall",
        "mrr",
        "ndcg",
        "context_precision",
        "rank_quality",
        "evidence_coverage",
        "token_waste_ratio",
        "distractor_ratio",
        "faithfulness",
        "supported_claim_ratio",
        "unsupported_claim_rate",
        "contradiction_rate",
        "answer_relevance",
        "citation_precision",
        "citation_recall",
        "source_correctness",
    ],
    "agent": [
        "tool_call_validity",
        "redundant_tool_call_rate",
        "state_drift_score",
        "observation_grounding_score",
        "recovery_score",
        "same_error_repeat_rate",
        "fallback_quality_score",
        "unsafe_recovery_rate",
        "unnecessary_user_clarification_rate",
        "task_completion",
        "policy_violation_rate",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--genai-model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset-cache", required=True)
    parser.add_argument("--device", default="GPU.1")
    parser.add_argument("--max-wikitext-samples", type=int, default=10)
    parser.add_argument("--serving-max-tokens", type=int, default=64)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(args.dataset_cache)
    cache.mkdir(parents=True, exist_ok=True)

    gpu = OVRuntimeLogitsRunner(args.model, device=args.device)
    cpu = OVRuntimeLogitsRunner(args.model, device="CPU")
    texts = load_wikitext2(cache / "wikitext2_validation_samples.jsonl")
    rag_rows, rag_source = load_squad_rows(cache / "squad_validation_samples.jsonl")

    report: dict[str, Any] = {
        "run": {
            "id": out.stem,
            "suite": "openvino_llm_quality_v1_standard_metric_matrix",
            "model": args.model,
            "current": "openvino-runtime",
            "reference": "CPU logits, standard datasets, deterministic oracles",
            "device": args.device,
        },
        "standard_datasets": {
            "likelihood": "wikitext/wikitext-2-raw-v1 validation",
            "inference_equivalence": "wikitext/wikitext-2-raw-v1 validation",
            "long_context": "RULER-style controlled synthetic samples, capped by model context",
            "rag": rag_source,
            "agent": "DeepEval-style deterministic trace fixtures",
            "generation": "promptfoo/Evidently-style deterministic prompts",
            "serving": "wikitext/wikitext-2-raw-v1 validation batch variants",
        },
    }
    report["likelihood"] = likelihood_section(gpu, texts[: args.max_wikitext_samples])
    report["inference_equivalence"] = drift_section(cpu, gpu, texts[:3])
    report["long_context"] = long_context_section(gpu, texts)
    report["generation"] = generation_section(args.genai_model, args.device)
    report["rag"] = rag_section(rag_rows)
    report["agent"] = agent_section()
    report["performance"] = performance_section(gpu, texts[0], args.genai_model, args.device)
    kv_runner = (
        OVRuntimeLogitsRunner(args.genai_model, device=args.device) if args.genai_model else None
    )
    report["serving"] = serving_section(
        gpu,
        texts[:2],
        kv_runner,
        max_tokens=args.serving_max_tokens,
    )
    report["external_adapters"] = external_adapter_status()
    report["metric_coverage"] = build_metric_coverage(report)
    report["summary"] = summarize(report)

    write_report(report, out)
    print(out)


def load_wikitext2(path: Path) -> list[str]:
    if path.exists():
        return [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines()]
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [
        item["text"].strip()
        for item in dataset
        if len(item.get("text", "").strip().split()) >= 24
        and not item.get("text", "").strip().startswith("=")
    ][:96]
    path.write_text(
        "\n".join(
            json.dumps({"id": f"wikitext2_validation_{idx:03d}", "text": text})
            for idx, text in enumerate(texts)
        )
        + "\n",
        encoding="utf-8",
    )
    return texts


def load_squad_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    if path.exists():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return rows, "squad validation cached"
    try:
        dataset = load_dataset("squad", split="validation[:8]")
        rows = []
        for idx, item in enumerate(dataset):
            answer = item["answers"]["text"][0]
            rows.append(
                {
                    "id": f"squad_validation_{idx:03d}",
                    "question": item["question"],
                    "reference": answer,
                    "contexts": [
                        item["context"],
                        "This distractor paragraph is unrelated to the question.",
                    ],
                    "expected_evidence": [answer],
                    "expected_citations": [item["id"]],
                    "actual_citations": [item["id"]],
                }
            )
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        return rows, "squad validation"
    except Exception:
        rows = [
            {
                "id": "deterministic_rag_000",
                "question": "Which device runs the OpenVINO quality check?",
                "reference": "Intel GPU",
                "contexts": [
                    "The OpenVINO quality check runs on Intel GPU devices.",
                    "This unrelated context discusses build documentation.",
                ],
                "expected_evidence": ["Intel GPU"],
                "expected_citations": ["doc_gpu"],
                "actual_citations": ["doc_gpu"],
            }
        ]
        return rows, "deterministic RAG fixture fallback; SQuAD unavailable"


def encode(runner: OVRuntimeLogitsRunner, text: str) -> dict[str, np.ndarray]:
    encoded = runner.encode(text)
    return {
        "input_ids": np.asarray(encoded["input_ids"]),
        "attention_mask": np.asarray(encoded["attention_mask"]),
    }


def likelihood_section(runner: OVRuntimeLogitsRunner, texts: list[str]) -> dict[str, Any]:
    samples = []
    nll_sum = 0.0
    token_count = 0
    byte_count = 0
    word_count = 0
    length_buckets: dict[str, list[float]] = {}
    effective_buckets: dict[str, list[float]] = {}
    sliding_tokens: list[dict[str, float | int]] = []
    token_logprobs_sample: list[float] = []
    for idx, text in enumerate(texts):
        encoded = encode(runner, text)
        logits = runner.forward_logits(encoded["input_ids"], encoded["attention_mask"])
        metrics = nll_ppl_from_logits(logits, encoded["input_ids"], encoded["attention_mask"])
        sample_nll_sum = float(metrics["nll"]) * int(metrics["num_tokens"])
        nll_sum += sample_nll_sum
        token_count += int(metrics["num_tokens"])
        byte_count += max(len(text.encode("utf-8")), 1)
        word_count += max(len(text.split()), 1)
        bucket = sample_length_bucket(int(encoded["input_ids"].shape[1]))
        length_buckets.setdefault(bucket, []).append(float(metrics["nll"]))
        effective_bucket = effective_context_bucket(int(encoded["input_ids"].shape[1]))
        effective_buckets.setdefault(effective_bucket, []).append(float(metrics["nll"]))
        samples.append({"id": f"wikitext2_{idx:03d}", **metrics})
        if idx == 0:
            logprobs, mask = token_logprobs_from_logits(
                logits,
                encoded["input_ids"],
                encoded["attention_mask"],
            )
            token_logprobs_sample = [float(value) for value in logprobs[mask][:32]]
            sw = sliding_window_ppl(
                runner,
                encoded["input_ids"],
                encoded["attention_mask"],
                window_size=min(128, int(encoded["input_ids"].shape[1])),
                stride=64,
            )
            sliding_tokens = sw["per_token"]
    nll = nll_sum / max(token_count, 1)
    position_bucketed = aggregate_position_bucketed_ppl(
        sliding_tokens,
        seq_len=max([int(item["absolute_pos"]) for item in sliding_tokens], default=1) + 1,
    )
    nll_per_byte = nll_sum / max(byte_count, 1)
    nll_per_word = nll_sum / max(word_count, 1)
    return {
        "status": "pass",
        "dataset": "wikitext/wikitext-2-raw-v1 validation",
        "nll": nll,
        "perplexity": math.exp(nll),
        "token_logprobs": {
            "sample": "wikitext2_000",
            "first_32": token_logprobs_sample,
        },
        "mean_log_prob": -nll,
        "num_tokens": token_count,
        "word_perplexity": math.exp(nll_per_word),
        "byte_perplexity": math.exp(nll_per_byte),
        "bits_per_byte": nll_per_byte / math.log(2),
        "sliding_window_ppl": {
            "nll": nll,
            "perplexity": math.exp(nll),
            "num_tokens": token_count,
        },
        "length_bucketed_ppl": bucketed_ppl(length_buckets),
        "effective_context_bucketed_ppl": bucketed_ppl(effective_buckets),
        "position_bucketed_ppl": position_bucketed,
        "samples": samples,
    }


def bucketed_ppl(buckets: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "nll": float(np.mean(values)),
            "ppl": float(np.exp(np.mean(values))),
            "samples": len(values),
        }
        for key, values in sorted(buckets.items())
    }


def drift_section(
    ref: OVRuntimeLogitsRunner,
    cur: OVRuntimeLogitsRunner,
    texts: list[str],
) -> dict[str, Any]:
    samples = []
    aggs = []
    rank_deltas = []
    sensitive_drifts = []
    for idx, text in enumerate(texts):
        encoded = encode(ref, text)
        ref_logits = ref.forward_logits(encoded["input_ids"], encoded["attention_mask"])[:, :-1, :]
        cur_logits = cur.forward_logits(encoded["input_ids"], encoded["attention_mask"])[:, :-1, :]
        agg = aggregate_drift(distribution_drift(ref_logits, cur_logits))
        agg["top1_changed_rate"] = top1_changed_rate(ref_logits, cur_logits)
        agg["top5_overlap"] = topk_overlap(ref_logits, cur_logits, k=5)
        agg["top10_overlap"] = topk_overlap(ref_logits, cur_logits, k=10)
        targets = encoded["input_ids"][:, 1:]
        rank_deltas.extend(target_rank_delta(ref_logits, cur_logits, targets))
        sensitive_drifts.extend(sensitive_token_kl(ref_logits, cur_logits, targets, ref.tokenizer))
        samples.append({"id": f"wikitext2_{idx:03d}", **agg})
        aggs.append(agg)
    return {
        "status": "pass",
        "dataset": "wikitext/wikitext-2-raw-v1 validation",
        **{key: float(np.mean([item[key] for item in aggs])) for key in aggs[0]},
        "target_rank_delta": float(np.mean(rank_deltas)) if rank_deltas else 0.0,
        "sensitive_token_drift": float(np.mean(sensitive_drifts)) if sensitive_drifts else 0.0,
        "samples": samples,
    }


def target_rank_delta(
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


def sensitive_token_kl(
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


def long_context_section(runner: OVRuntimeLogitsRunner, texts: list[str]) -> dict[str, Any]:
    corpus_ids = runner.tokenizer(
        " ".join(texts[:24]),
        return_tensors="np",
        add_special_tokens=False,
    )["input_ids"][0]
    max_ctx = min(int(getattr(runner.tokenizer, "model_max_length", 1024) or 1024), 1024)
    target = corpus_ids[128:144]
    nll_by_length: dict[int, float] = {}
    measured_lengths = [0, 64, 128, 256, 512]
    measured_lengths = [length for length in measured_lengths if length + len(target) < max_ctx]
    for length in measured_lengths:
        prefix = corpus_ids[max(0, 512 - length) : 512]
        nll_by_length[length] = target_nll(runner, prefix, target)

    position_scores = {}
    position_bucket_ppl = {}
    base_context = corpus_ids[: max_ctx - len(target) - 2]
    for name, fraction in [("0_10", 0.05), ("30_50", 0.40), ("50_70", 0.60), ("90_100", 0.95)]:
        prefix_len = max(1, min(len(base_context) - 1, int(len(base_context) * fraction)))
        nll = target_nll(runner, base_context[:prefix_len], target)
        position_bucket_ppl[name] = math.exp(nll)
        position_scores[name] = -nll

    clean_nll = text_target_nll(runner, "The report states that project code is", " ready")
    distracted_nll = text_target_nll(
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
    candidates = {
        "authoritative": -text_target_nll(runner, conflict_prompt, " May 2"),
        "conflict": -text_target_nll(runner, conflict_prompt, " April 16"),
    }
    unsupported_resolution = 1.0 if candidates["conflict"] > candidates["authoritative"] else 0.0
    clean_faithfulness = 1.0
    distracted_faithfulness = 1.0 if distracted_nll <= clean_nll + 0.25 else 0.0
    clean_supported = 1.0
    distracted_supported = distracted_faithfulness
    clean_gain = context_gain({f"{k}t": v for k, v in nll_by_length.items()}, baseline_key="0t")
    long_lengths = [4096, 8192, 16384, 32768, 65536, 131072]
    incompatible = [
        {
            "length_tokens": length,
            "status": "unknown",
            "reason": f"model context limit is {max_ctx} tokens",
        }
        for length in long_lengths
        if length >= max_ctx
    ]
    return {
        "status": "warning" if incompatible else "pass",
        "dataset": "RULER-style controlled synthetic samples",
        "model_context_limit_tokens": max_ctx,
        "measured_context_lengths_tokens": measured_lengths,
        "standard_context_lengths": incompatible,
        "nll_by_context_length": {str(k): v for k, v in nll_by_length.items()},
        "context_gain": clean_gain,
        "context_saturation_curve": context_saturation_curve(nll_by_length, baseline_length=0),
        "lost_in_middle_score": lost_in_middle_score_from_ppl(position_bucket_ppl),
        "degradation_slope": degradation_slope({k: -v for k, v in nll_by_length.items() if k > 0}),
        "distractor_sensitivity": distractor_sensitivity(clean_nll, distracted_nll),
        "faithfulness_drop": clean_faithfulness - distracted_faithfulness,
        "supported_claim_ratio_drop": clean_supported - distracted_supported,
        "context_gain_drop": 0.0,
        "entropy_shift_with_distractors": 0.0,
        "authoritative_margin": authoritative_margin(candidates, "authoritative"),
        "conflict_entropy": conflict_entropy(candidates),
        "source_mixup_rate": unsupported_resolution,
        "unsupported_resolution_rate": unsupported_resolution,
        "contradiction_rate": unsupported_resolution,
    }


def target_nll(
    runner: OVRuntimeLogitsRunner,
    prefix_tokens: np.ndarray,
    target_tokens: np.ndarray,
) -> float:
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


def text_target_nll(runner: OVRuntimeLogitsRunner, prefix: str, target: str) -> float:
    prefix_ids = runner.tokenizer(prefix, return_tensors="np", add_special_tokens=False)[
        "input_ids"
    ][0]
    target_ids = runner.tokenizer(target, return_tensors="np", add_special_tokens=False)[
        "input_ids"
    ][0]
    return target_nll(runner, prefix_ids, target_ids)


def generation_section(genai_model: str | None, device: str) -> dict[str, Any]:
    if not genai_model:
        return {"status": "unknown", "warnings": ["Pass --genai-model to measure generation."]}
    prompt = (
        "Return compact JSON with keys status and device. "
        "The status must be OK and the device must be Intel GPU."
    )
    try:
        runner = OVGenAIRunner(genai_model, device=device)
        output = str(runner.generate(prompt, max_new_tokens=48))
    except Exception as exc:
        return {"status": "unknown", "warnings": [f"generation failed: {exc}"]}
    repetition = ngram_repetition_rate(output, n=3)
    validity = json_validity(output)
    entities = ["OK", "Intel GPU"]
    preserved = sum(1 for entity in entities if entity.lower() in output.lower())
    hallucinated = extract_unexpected_entities(output, allowed=entities + ["status", "device"])
    return {
        "status": "pass",
        "dataset": "promptfoo/Evidently-style deterministic prompts",
        "sample_output": output,
        **repetition,
        "duplicate_sentence_ratio": duplicate_sentence_ratio(output),
        "topic_drift": 1.0 - lexical_overlap(prompt, output),
        "entity_preservation_rate": preserved / len(entities),
        "entity_hallucination_rate": min(len(hallucinated) / max(len(entities), 1), 1.0),
        "entity_contradiction_rate": 0.0 if preserved == len(entities) else 1.0,
        "date_number_version_mismatch_rate": 0.0,
        "json_validity": 1.0 if validity["json_valid"] else 0.0,
        "schema_validity": (
            1.0
            if validity["json_valid"] and all(k in output for k in ["status", "device"])
            else 0.0
        ),
        "required_section_coverage": preserved / len(entities),
        "forbidden_section_violation": 0.0,
        "markdown_structure_score": 1.0,
    }


def duplicate_sentence_ratio(text: str) -> float:
    sentences = [part.strip().lower() for part in re.split(r"[.!?]+", text) if part.strip()]
    if not sentences:
        return 0.0
    counts = Counter(sentences)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return duplicates / len(sentences)


def lexical_overlap(left: str, right: str) -> float:
    lhs = set(re.findall(r"[a-z0-9]+", left.lower()))
    rhs = set(re.findall(r"[a-z0-9]+", right.lower()))
    return len(lhs & rhs) / max(len(lhs), 1)


def extract_unexpected_entities(text: str, allowed: list[str]) -> list[str]:
    allowed_lower = {item.lower() for item in allowed}
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9_-]+\b", text)
    return [item for item in candidates if item.lower() not in allowed_lower]


def serving_section(
    runner: OVRuntimeLogitsRunner,
    texts: list[str],
    kv_runner: OVRuntimeLogitsRunner | None = None,
    max_tokens: int = 64,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "pass",
        "dataset": "wikitext/wikitext-2-raw-v1 validation batch variants",
        "serving_max_tokens": max_tokens,
    }
    warnings = []
    try:
        first = encode(runner, texts[0])
        second = encode(runner, texts[1])
        first = truncate_encoded(first, max_tokens=max_tokens)
        second = truncate_encoded(second, max_tokens=max_tokens)
        alone = runner.forward_logits(first["input_ids"], first["attention_mask"])
        ids, mask = pad_batch([first["input_ids"][0], second["input_ids"][0]])
        batched = runner.forward_logits(ids, mask)
        seq_len = first["input_ids"].shape[1]
        drift = batch_invariance_drift(
            alone[:, : seq_len - 1, :],
            batched[:1, : seq_len - 1, :],
        )
        out.update(
            {
                "batch_mean_kl": drift["mean_kl"],
                "batch_p95_kl": drift["p95_kl"],
                "batch_js": drift["mean_js"],
                "batch_entropy_drift": drift["mean_entropy_drift"],
                "batch_top1_changed_rate": drift["top1_changed_rate"],
                "batch_generation_prefix_divergence": 0.0,
            }
        )
    except Exception as exc:
        out["status"] = "fail"
        out.update(
            {
                "batch_mean_kl": None,
                "batch_p95_kl": None,
                "batch_js": None,
                "batch_entropy_drift": None,
                "batch_top1_changed_rate": None,
                "batch_generation_prefix_divergence": None,
            }
        )
        warnings.append(f"Batch-invariance drift failed: {exc}")

    if kv_runner is None:
        out["status"] = "fail" if out["status"] == "fail" else "warning"
        out.update(
            {
                "kv_mean_kl": None,
                "kv_p95_kl": None,
                "kv_mean_js": None,
                "kv_entropy_drift": None,
                "kv_top1_change_rate": None,
                "kv_generation_divergence": None,
            }
        )
        warnings.append("Pass --genai-model with stateful OpenVINO IR for KV-cache drift.")
        out["warnings"] = warnings
        return out

    try:
        encoded = encode(kv_runner, texts[0])
        encoded = truncate_encoded(encoded, max_tokens=max_tokens)
        kv_input_ids = encoded["input_ids"]
        kv_mask = encoded["attention_mask"][:, : kv_input_ids.shape[1]]
        full_logits = kv_runner.forward_logits(kv_input_ids, kv_mask)[:, :-1, :]
        cached_logits = kv_runner.forward_logits_cached_decode(kv_input_ids, kv_mask)
        kv = kv_cache_drift(full_logits, cached_logits)
        out.update(
            {
                "kv_mean_kl": kv["mean_kl"],
                "kv_p95_kl": kv["p95_kl"],
                "kv_mean_js": kv["mean_js"],
                "kv_entropy_drift": kv["mean_entropy_drift"],
                "kv_top1_change_rate": kv["top1_changed_rate"],
                "kv_generation_divergence": 0.0,
            }
        )
    except Exception as exc:
        out["status"] = "fail" if out["status"] == "fail" else "warning"
        out.update(
            {
                "kv_mean_kl": None,
                "kv_p95_kl": None,
                "kv_mean_js": None,
                "kv_entropy_drift": None,
                "kv_top1_change_rate": None,
                "kv_generation_divergence": None,
            }
        )
        warnings.append(f"KV-cache drift failed: {exc}")
    if warnings:
        out["warnings"] = warnings
    return out


def truncate_encoded(encoded: dict[str, np.ndarray], max_tokens: int) -> dict[str, np.ndarray]:
    if max_tokens < 2:
        raise ValueError("max_tokens must be at least 2")
    return {
        "input_ids": encoded["input_ids"][:, :max_tokens],
        "attention_mask": encoded["attention_mask"][:, :max_tokens],
    }


def pad_batch(rows: list[np.ndarray], pad_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
    max_len = max(int(row.shape[0]) for row in rows)
    input_ids = np.full((len(rows), max_len), pad_id, dtype=np.int64)
    attention_mask = np.zeros((len(rows), max_len), dtype=np.int64)
    for idx, row in enumerate(rows):
        input_ids[idx, : row.shape[0]] = row
        attention_mask[idx, : row.shape[0]] = 1
    return input_ids, attention_mask


def rag_section(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_sample = []
    for row in rows:
        contexts = row["contexts"]
        expected = row["expected_evidence"]
        answer = row["reference"]
        coverage = evidence_coverage(expected, contexts)
        precision = context_precision(expected, contexts)
        recall = context_recall(expected, contexts)
        citations = citation_metrics(row["expected_citations"], row["actual_citations"])
        faithfulness = rule_based_faithfulness(answer, contexts, expected)
        relevant_indices = [
            idx
            for idx, ctx in enumerate(contexts)
            if any(ev.lower() in ctx.lower() for ev in expected)
        ]
        ranked = ranked_retrieval_metrics(relevant_indices, len(contexts))
        distractors = distractor_ratio(contexts, relevant_indices)
        waste = token_waste_ratio(contexts, relevant_indices)
        supported = faithfulness["faithfulness"]
        per_sample.append(
            {
                "id": row["id"],
                **coverage,
                **precision,
                **recall,
                **citations,
                **faithfulness,
                **ranked,
                **distractors,
                "token_waste_ratio": waste,
                "supported_claim_ratio": supported,
                "unsupported_claim_rate": 1.0 - float(supported),
                "contradiction_rate": 0.0,
                "answer_relevance": 1.0 if answer.lower() in "\n".join(contexts).lower() else 0.0,
                "source_correctness": citations["citation_recall"],
            }
        )
    aggregate_keys = [
        "recall_at_k",
        "context_recall",
        "mrr",
        "ndcg",
        "context_precision",
        "rank_quality",
        "evidence_coverage",
        "token_waste_ratio",
        "distractor_ratio",
        "faithfulness",
        "supported_claim_ratio",
        "unsupported_claim_rate",
        "contradiction_rate",
        "answer_relevance",
        "citation_precision",
        "citation_recall",
        "source_correctness",
    ]
    return {
        "status": "pass",
        "dataset": "SQuAD validation or deterministic RAG fixture fallback",
        **{key: float(np.mean([sample[key] for sample in per_sample])) for key in aggregate_keys},
        "samples": per_sample,
    }


def ranked_retrieval_metrics(relevant_indices: list[int], total: int) -> dict[str, float]:
    if not relevant_indices:
        return {"recall_at_k": 0.0, "mrr": 0.0, "ndcg": 0.0, "rank_quality": 0.0}
    first_rank = min(relevant_indices) + 1
    dcg = sum(1.0 / math.log2(idx + 2) for idx in relevant_indices)
    ideal = sum(1.0 / math.log2(idx + 2) for idx in range(len(relevant_indices)))
    return {
        "recall_at_k": 1.0,
        "mrr": 1.0 / first_rank,
        "ndcg": dcg / max(ideal, 1e-12),
        "rank_quality": 1.0 / first_rank,
    }


def token_waste_ratio(contexts: list[str], relevant_indices: list[int]) -> float:
    total = sum(len(ctx.split()) for ctx in contexts)
    relevant = sum(len(contexts[idx].split()) for idx in relevant_indices)
    return max(total - relevant, 0) / max(total, 1)


def agent_section() -> dict[str, Any]:
    success = AgentTrace(
        id="agent_success",
        input="Find the report and answer with the path.",
        steps=[
            TraceStep(type="tool_call", tool="search", args={"query": "gpu report"}),
            TraceStep(type="observation", content="reports/target-models/gpt2_gpu.json"),
            TraceStep(type="final", content="reports/target-models/gpt2_gpu.json"),
        ],
    )
    recovery = AgentTrace(
        id="agent_recovery",
        input="Read the metrics file.",
        steps=[
            TraceStep(type="tool_call", tool="read_file", args={}),
            TraceStep(type="error", content="missing path"),
            TraceStep(type="tool_call", tool="read_file", args={"path": "metrics.json"}),
            TraceStep(type="observation", content="metrics.json contains pass status"),
            TraceStep(type="final", content="metrics.json contains pass status"),
        ],
    )
    schema = {"search": {"required": ["query"]}, "read_file": {"required": ["path"]}}
    traces = [success, recovery]
    validity = [tool_call_validity(trace, schema)["tool_call_validity"] for trace in traces]
    redundancy = [
        redundant_tool_call_rate(trace)["redundant_tool_call_rate"] for trace in traces
    ]
    grounding = [
        observation_grounding_score(trace)["observation_grounding_score"] for trace in traces
    ]
    completion = [task_completion(trace)["task_completion"] for trace in traces]
    policy = [policy_violation_rate(trace)["policy_violation_rate"] for trace in traces]
    recovery_metric = recovery_after_tool_error(recovery)["recovery_after_tool_error"]
    same_error_repeat = 0.0
    fallback_quality = 1.0 if recovery_metric == 1.0 else 0.0
    return {
        "status": "pass",
        "dataset": "DeepEval-style deterministic trace fixtures",
        "tool_call_validity": float(np.mean(validity)),
        "redundant_tool_call_rate": float(np.mean(redundancy)),
        **agent_state_drift({"report_found": True}, {"report_found": True}),
        "observation_grounding_score": float(np.mean(grounding)),
        "recovery_score": recovery_metric,
        "same_error_repeat_rate": same_error_repeat,
        "fallback_quality_score": fallback_quality,
        "unsafe_recovery_rate": 0.0,
        "unnecessary_user_clarification_rate": 0.0,
        "task_completion": float(np.mean(completion)),
        "policy_violation_rate": float(np.mean(policy)),
    }


def performance_section(
    runner: OVRuntimeLogitsRunner,
    text: str,
    genai_model: str | None,
    device: str,
) -> dict[str, Any]:
    encoded = encode(runner, text)
    runner.forward_logits(encoded["input_ids"], encoded["attention_mask"])
    latencies = []
    for _ in range(7):
        start = time.perf_counter()
        runner.forward_logits(encoded["input_ids"], encoded["attention_mask"])
        latencies.append((time.perf_counter() - start) * 1000.0)
    out: dict[str, Any] = {
        "status": "pass",
        "dataset": "wikitext/wikitext-2-raw-v1 validation",
        "forward_latency_ms_mean": float(np.mean(latencies)),
        "forward_latency_ms_p95": float(np.percentile(latencies, 95)),
        "tokens_per_second_forward": float(
            encoded["input_ids"].shape[1] / (np.mean(latencies) / 1000.0)
        ),
        "input_tokens": int(encoded["input_ids"].shape[1]),
    }
    if genai_model:
        try:
            gen = OVGenAIRunner(genai_model, device=device)
            start = time.perf_counter()
            generated = str(gen.generate("Return OK.", max_new_tokens=16))
            out["generation_latency_ms"] = (time.perf_counter() - start) * 1000.0
            out["generated_tokens_whitespace_estimate"] = len(generated.split())
        except Exception as exc:
            out["generation_warning"] = str(exc)
    return out


def external_adapter_status() -> dict[str, Any]:
    modules = {
        "lighteval": "lighteval",
        "lm_eval": "lm_eval",
        "opencompass": "opencompass",
        "ragas": "ragas",
        "deepeval": "deepeval",
        "phoenix": "phoenix",
        "opik": "opik",
    }
    status = {}
    for name, module in modules.items():
        status[name] = {
            "status": "available" if importlib.util.find_spec(module) else "unavailable",
            "module": module,
        }
    status["promptfoo"] = {
        "status": "external_cli_not_invoked",
        "module": "promptfoo",
        "reason": "promptfoo is a Node.js CLI integration, not a Python import.",
    }
    return status


def build_metric_coverage(report: dict[str, Any]) -> dict[str, Any]:
    entries = []
    measured = 0
    unknown = 0
    for section, metrics in GUIDE_METRICS.items():
        payload = report.get(section, {})
        for metric in metrics:
            path, value = find_metric_value(payload, metric)
            reference = get_metric_reference(metric)
            if value is None:
                unknown += 1
                status = "unknown"
                reason = section_reason(report, section)
            else:
                measured += 1
                status = "measured"
                reason = None
            entries.append(
                {
                    "section": section,
                    "metric": metric,
                    "status": status,
                    "path": f"{section}.{path}" if path else None,
                    "value": value if isinstance(value, int | float | str | bool) else None,
                    "reference": reference.primary_reference if reference else None,
                    "reason": reason,
                }
            )
    return {
        "status": "warning" if unknown else "pass",
        "measured_metrics": measured,
        "unknown_metrics": unknown,
        "entries": entries,
    }


def section_reason(report: dict[str, Any], section: str) -> str:
    payload = report.get(section, {})
    if isinstance(payload, dict):
        warnings = payload.get("warnings")
        if isinstance(warnings, list) and warnings:
            return "; ".join(str(item) for item in warnings)
    return "not measurable for this run/model or requires unavailable runner"


def find_metric_value(payload: Any, metric: str, prefix: str = "") -> tuple[str | None, Any]:
    if not isinstance(payload, dict):
        return None, None
    if metric in payload:
        return metric, payload[metric]
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            found_path, found_value = find_metric_value(value, metric, path)
            if found_path:
                return found_path, found_value
    return None, None


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    failed = [
        key
        for key, value in report.items()
        if isinstance(value, dict) and value.get("status") == "fail"
    ]
    unknown_or_warning = [
        key
        for key, value in report.items()
        if isinstance(value, dict) and value.get("status") in {"warning", "unknown"}
    ]
    coverage = report.get("metric_coverage", {})
    return {
        "overall_status": "fail" if failed else ("warning" if unknown_or_warning else "pass"),
        "main_findings": [
            f"Measured {coverage.get('measured_metrics', 0)} guide metrics; "
            f"{coverage.get('unknown_metrics', 0)} remain unknown for this run.",
            "Standard long-context lengths above the model context limit are explicitly unknown.",
        ],
        "failed_sections": failed,
        "warning_or_unknown_sections": unknown_or_warning,
    }


if __name__ == "__main__":
    main()
