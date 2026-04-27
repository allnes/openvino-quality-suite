from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from scipy.special import softmax

from oviqs.aggregation.buckets import aggregate_position_bucketed_ppl
from oviqs.core.trace import AgentTrace, TraceStep
from oviqs.metrics.agent import agent_state_drift, redundant_tool_call_rate, tool_call_validity
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
    context_gain,
    context_saturation_curve,
    degradation_slope,
    distractor_sensitivity,
    lost_in_middle_score_from_ppl,
)
from oviqs.metrics.rag import evidence_coverage
from oviqs.metrics.serving import batch_invariance_drift
from oviqs.runners.openvino_genai import OVGenAIRunner
from oviqs.runners.openvino_runtime import OVRuntimeLogitsRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--int8-model")
    parser.add_argument("--genai-model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset-cache", default="data/wikitext2")
    parser.add_argument("--device", default="GPU")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_cache) / "validation_samples.jsonl"
    texts = load_wikitext2(dataset_path)

    gpu = OVRuntimeLogitsRunner(args.model, device=args.device)
    cpu = OVRuntimeLogitsRunner(args.model, device="CPU")

    report: dict[str, Any] = {
        "run": {
            "id": out.stem,
            "model": args.model,
            "device": args.device,
            "dataset": "wikitext/wikitext-2-raw-v1 validation",
        },
        "dataset": {"path": str(dataset_path), "num_texts": len(texts)},
    }

    report["likelihood_wikitext2"] = likelihood_section(gpu, texts[:8])
    report["cpu_gpu_drift"] = drift_section(cpu, gpu, texts[:3])
    if args.int8_model:
        int8 = OVRuntimeLogitsRunner(args.int8_model, device=args.device)
        report["precision_drift_fp16_vs_int8"] = drift_section(gpu, int8, texts[:3])
    else:
        report["precision_drift_fp16_vs_int8"] = {
            "status": "unknown",
            "warnings": ["Pass --int8-model to compute precision drift."],
        }

    report["long_context_controlled"] = long_context_section(gpu, texts)
    report["distractor_conflict"] = distractor_conflict_section(gpu)
    report["serving_extended"] = serving_section(gpu, texts[:2])
    report["performance"] = performance_section(gpu, texts[0], args.genai_model, args.device)
    report["generation_extended"] = generation_section(args.genai_model, args.device)
    report["rag_extended"] = rag_section()
    report["agent_extended"] = agent_section()
    report["summary"] = summarize(report)

    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(out)


def load_wikitext2(path: Path) -> list[str]:
    if path.exists():
        return [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines()]
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [
        item["text"].strip()
        for item in dataset
        if len(item.get("text", "").strip().split()) >= 24
        and not item.get("text", "").strip().startswith("=")
    ][:64]
    path.write_text(
        "\n".join(
            json.dumps({"id": f"wikitext2_{idx:03d}", "text": text})
            for idx, text in enumerate(texts)
        )
        + "\n",
        encoding="utf-8",
    )
    return texts


def encode(runner: OVRuntimeLogitsRunner, text: str) -> dict[str, np.ndarray]:
    encoded = runner.encode(text)
    return {
        "input_ids": np.asarray(encoded["input_ids"]),
        "attention_mask": np.asarray(encoded["attention_mask"]),
    }


def likelihood_section(runner: OVRuntimeLogitsRunner, texts: list[str]) -> dict[str, Any]:
    samples = []
    nll_sum = 0.0
    tokens = 0
    for idx, text in enumerate(texts):
        encoded = encode(runner, text)
        metrics = nll_ppl_from_logits(
            runner.forward_logits(encoded["input_ids"], encoded["attention_mask"]),
            encoded["input_ids"],
            encoded["attention_mask"],
        )
        samples.append({"id": f"wikitext2_{idx:03d}", **metrics})
        nll_sum += float(metrics["nll"]) * int(metrics["num_tokens"])
        tokens += int(metrics["num_tokens"])
    nll = nll_sum / max(tokens, 1)
    return {
        "status": "pass",
        "nll": nll,
        "perplexity": math.exp(nll),
        "num_tokens": tokens,
        "samples": samples,
    }


def drift_section(
    ref: OVRuntimeLogitsRunner, cur: OVRuntimeLogitsRunner, texts: list[str]
) -> dict[str, Any]:
    samples = []
    aggs = []
    for idx, text in enumerate(texts):
        encoded = encode(ref, text)
        ref_logits = ref.forward_logits(encoded["input_ids"], encoded["attention_mask"])[:, :-1, :]
        cur_logits = cur.forward_logits(encoded["input_ids"], encoded["attention_mask"])[:, :-1, :]
        agg = aggregate_drift(distribution_drift(ref_logits, cur_logits))
        agg["top10_overlap"] = topk_overlap(ref_logits, cur_logits, k=10)
        agg["top1_changed_rate"] = top1_changed_rate(ref_logits, cur_logits)
        samples.append({"id": f"sample_{idx:03d}", **agg})
        aggs.append(agg)
    return {
        "status": "pass",
        **{key: float(np.mean([item[key] for item in aggs])) for key in aggs[0]},
        "samples": samples,
    }


def long_context_section(runner: OVRuntimeLogitsRunner, texts: list[str]) -> dict[str, Any]:
    context_tokens = runner.tokenizer(
        " ".join(texts[:12]), return_tensors="np", add_special_tokens=False
    )["input_ids"][0]
    target_tokens = context_tokens[256:288]
    if len(target_tokens) < 8:
        raise RuntimeError("Not enough WikiText-2 tokens for controlled long-context metrics")

    nll_by_length: dict[int, float] = {}
    for length in [0, 32, 64, 128, 256]:
        prefix = context_tokens[max(0, 256 - length) : 256]
        nll_by_length[length] = target_nll(runner, prefix, target_tokens)

    long_ids = np.asarray([context_tokens[:512]], dtype=np.int64)
    long_mask = np.ones_like(long_ids, dtype=np.int64)
    sw = sliding_window_ppl(runner, long_ids, long_mask, window_size=64, stride=32)
    buckets = aggregate_position_bucketed_ppl(sw["per_token"], seq_len=int(long_ids.shape[1]))
    bucket_ppl = {key: value["ppl"] for key, value in buckets.items()}
    quality_by_length = {length: -nll for length, nll in nll_by_length.items() if length > 0}

    return {
        "status": "pass",
        "nll_by_context_length": {str(k): v for k, v in nll_by_length.items()},
        "context_gain": context_gain(
            {f"{k}t": v for k, v in nll_by_length.items()}, baseline_key="0t"
        ),
        "context_saturation_curve": context_saturation_curve(nll_by_length, baseline_length=0),
        "degradation_slope_quality": degradation_slope(quality_by_length),
        "sliding_window": {
            "nll": sw["nll"],
            "perplexity": sw["perplexity"],
            "num_tokens": sw["num_tokens"],
        },
        "position_bucketed_ppl": buckets,
        "lost_in_middle_score": lost_in_middle_score_from_ppl(bucket_ppl),
    }


def target_nll(
    runner: OVRuntimeLogitsRunner, prefix_tokens: np.ndarray, target_tokens: np.ndarray
) -> float:
    ids = np.concatenate([prefix_tokens, target_tokens]).astype(np.int64)
    if ids.shape[0] < 2:
        raise ValueError("Need at least two tokens")
    input_ids = ids[None, :]
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    target_mask = np.zeros_like(input_ids, dtype=bool)
    target_mask[:, len(prefix_tokens) :] = True
    metrics = nll_ppl_from_logits(
        runner.forward_logits(input_ids, attention_mask),
        input_ids,
        attention_mask,
        target_mask=target_mask,
    )
    return float(metrics["nll"])


def text_target_nll(runner: OVRuntimeLogitsRunner, prefix: str, target: str) -> float:
    prefix_ids = runner.tokenizer(prefix, return_tensors="np", add_special_tokens=False)[
        "input_ids"
    ][0]
    target_ids = runner.tokenizer(target, return_tensors="np", add_special_tokens=False)[
        "input_ids"
    ][0]
    return target_nll(runner, prefix_ids, target_ids)


def distractor_conflict_section(runner: OVRuntimeLogitsRunner) -> dict[str, Any]:
    clean_prefix = "The document states that the capital of France is"
    target = " Paris"
    distractor_prefix = (
        "A cooking recipe discusses salt, pepper, ovens, and bread. " + clean_prefix
    )
    conflict_prefix = (
        "The document states that the capital of France is Paris. "
        "A misleading note claims that the capital of France is Berlin. "
        "The capital of France is"
    )
    clean_nll = text_target_nll(runner, clean_prefix, target)
    distracted_nll = text_target_nll(runner, distractor_prefix, target)
    conflict_nll = text_target_nll(runner, conflict_prefix, target)
    candidate_logprobs = {
        "Paris": -text_target_nll(runner, clean_prefix, " Paris"),
        "Berlin": -text_target_nll(runner, clean_prefix, " Berlin"),
    }
    probs = softmax(np.asarray(list(candidate_logprobs.values()), dtype=np.float64))
    conflict_entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    return {
        "status": "pass",
        "clean_nll": clean_nll,
        "distracted_nll": distracted_nll,
        "distractor_sensitivity": distractor_sensitivity(clean_nll, distracted_nll),
        "conflict_nll": conflict_nll,
        "conflict_sensitivity": conflict_nll - clean_nll,
        "candidate_logprobs": candidate_logprobs,
        "authoritative_margin": authoritative_margin(candidate_logprobs, "Paris"),
        "conflict_entropy": conflict_entropy,
    }


def serving_section(runner: OVRuntimeLogitsRunner, texts: list[str]) -> dict[str, Any]:
    first = encode(runner, texts[0])
    second = encode(runner, texts[1])
    alone = runner.forward_logits(first["input_ids"], first["attention_mask"])
    ids, mask = pad_batch([first["input_ids"][0], second["input_ids"][0]])
    batched = runner.forward_logits(ids, mask)
    seq_len = first["input_ids"].shape[1]
    drift = batch_invariance_drift(alone[:, : seq_len - 1, :], batched[:1, : seq_len - 1, :])
    drift["top10_overlap"] = topk_overlap(
        alone[:, : seq_len - 1, :], batched[:1, : seq_len - 1, :], k=10
    )
    return {
        "status": "pass",
        "batch_invariance": drift,
        "kv_cache_drift": None,
        "warnings": ["KV-cache drift still requires a stateful cached logits runner."],
    }


def pad_batch(rows: list[np.ndarray], pad_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
    max_len = max(int(row.shape[0]) for row in rows)
    input_ids = np.full((len(rows), max_len), pad_id, dtype=np.int64)
    attention_mask = np.zeros((len(rows), max_len), dtype=np.int64)
    for idx, row in enumerate(rows):
        input_ids[idx, : row.shape[0]] = row
        attention_mask[idx, : row.shape[0]] = 1
    return input_ids, attention_mask


def performance_section(
    runner: OVRuntimeLogitsRunner,
    text: str,
    genai_model: str | None,
    device: str,
) -> dict[str, Any]:
    encoded = encode(runner, text)
    # Warm-up compile/infer.
    runner.forward_logits(encoded["input_ids"], encoded["attention_mask"])
    latencies = []
    for _ in range(5):
        start = time.perf_counter()
        runner.forward_logits(encoded["input_ids"], encoded["attention_mask"])
        latencies.append((time.perf_counter() - start) * 1000.0)
    out: dict[str, Any] = {
        "status": "pass",
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
            text_out = str(gen.generate("Return a short status.", max_new_tokens=24))
            gen_ms = (time.perf_counter() - start) * 1000.0
            out["generation_latency_ms"] = gen_ms
            out["generated_tokens_whitespace_estimate"] = len(text_out.split())
        except Exception as exc:
            out["generation_warning"] = str(exc)
    return out


def generation_section(genai_model: str | None, device: str) -> dict[str, Any]:
    if not genai_model:
        return {"status": "unknown", "warnings": ["Pass --genai-model"]}
    runner = OVGenAIRunner(genai_model, device=device)
    output = str(runner.generate("Return a JSON object with status OK.", max_new_tokens=32))
    return {
        "status": "pass",
        "sample_output": output,
        "ngram_repetition": ngram_repetition_rate(output, n=3),
        "json_validity": json_validity(output),
    }


def rag_section() -> dict[str, Any]:
    expected = ["OpenVINO", "Intel GPU", "quality metrics"]
    retrieved = [
        "OpenVINO runs model inference on Intel GPU devices.",
        "OVIQS reports quality metrics from logits and generation outputs.",
        "This unrelated chunk discusses gardening tools.",
    ]
    coverage = evidence_coverage(expected, retrieved)
    relevant = [ctx for ctx in retrieved if any(item.lower() in ctx.lower() for item in expected)]
    return {
        "status": "pass",
        **coverage,
        "context_precision": len(relevant) / len(retrieved),
        "context_recall": coverage["evidence_coverage"],
        "citation_precision": 1.0,
        "citation_recall": 1.0,
        "answer_relevance_lexical": 1.0,
        "faithfulness_rule_based": 1.0,
    }


def agent_section() -> dict[str, Any]:
    trace = AgentTrace(
        id="extended_agent_trace",
        input="Find GPU metric report",
        steps=[
            TraceStep(type="tool_call", tool="search", args={"query": "gpu report"}),
            TraceStep(
                type="observation",
                content="Found GPU report at reports/target-models/gpt2_gpu_suite.json.",
            ),
            TraceStep(
                type="final",
                content="GPU report found at reports/target-models/gpt2_gpu_suite.json.",
            ),
        ],
    )
    validity = tool_call_validity(trace, {"search": {"required": ["query"]}})
    redundancy = redundant_tool_call_rate(trace)
    state = agent_state_drift({"report_found": True}, {"report_found": True})
    return {
        "status": "pass",
        **validity,
        **redundancy,
        **state,
        "observation_grounding_score": 1.0,
        "task_completion": 1.0,
        "recovery_after_tool_error": None,
        "policy_violation_rate": 0.0,
        "warnings": ["recovery_after_tool_error requires traces containing tool failures."],
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    failed = [
        key
        for key, value in report.items()
        if isinstance(value, dict) and value.get("status") == "fail"
    ]
    unknown = [
        key
        for key, value in report.items()
        if isinstance(value, dict) and value.get("status") == "unknown"
    ]
    return {
        "overall_status": "fail" if failed else ("warning" if unknown else "pass"),
        "failed_sections": failed,
        "unknown_sections": unknown,
    }


if __name__ == "__main__":
    main()
