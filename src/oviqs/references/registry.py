from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ReferenceKind = Literal[
    "external_framework",
    "official_formula",
    "deterministic_golden_fixture",
    "human_or_judge_label",
]


@dataclass(frozen=True)
class ReferenceSource:
    """A concrete source that defines expected metric behavior."""

    name: str
    kind: ReferenceKind
    url: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {"name": self.name, "kind": self.kind}
        if self.url:
            payload["url"] = self.url
        if self.notes:
            payload["notes"] = self.notes
        return payload


@dataclass(frozen=True)
class MetricReference:
    """Reference/oracle metadata required before trusting an OVIQS metric."""

    family: str
    metric_names: tuple[str, ...]
    primary_reference: str
    sources: tuple[ReferenceSource, ...]
    oracle: str
    degradation_rule: str
    required_inputs: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "metric_names": list(self.metric_names),
            "primary_reference": self.primary_reference,
            "sources": [source.to_dict() for source in self.sources],
            "oracle": self.oracle,
            "degradation_rule": self.degradation_rule,
            "required_inputs": list(self.required_inputs),
        }


LM_EVAL = ReferenceSource(
    name="EleutherAI lm-evaluation-harness",
    kind="external_framework",
    url="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md",
    notes="Reference for loglikelihood and loglikelihood_rolling semantics.",
)
LIGHTEVAL = ReferenceSource(
    name="Hugging Face LightEval",
    kind="external_framework",
    url="https://huggingface.co/docs/lighteval/package_reference/metrics",
    notes="Reference for corpus-level perplexity metric structure.",
)
HF_PPL = ReferenceSource(
    name="Hugging Face Transformers perplexity guide",
    kind="official_formula",
    url="https://huggingface.co/docs/transformers/en/perplexity",
    notes="Reference for fixed-length/sliding-window perplexity evaluation.",
)
SCIPY_ENTROPY = ReferenceSource(
    name="SciPy entropy/KL",
    kind="official_formula",
    url="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html",
    notes="Reference for Shannon entropy and KL divergence formulas.",
)
SCIKIT_NDCG = ReferenceSource(
    name="scikit-learn ndcg_score",
    kind="official_formula",
    url="https://sklearn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html",
    notes="Reference for ranked retrieval nDCG behavior.",
)
SENTENCE_TRANSFORMERS_IR = ReferenceSource(
    name="SentenceTransformers InformationRetrievalEvaluator",
    kind="external_framework",
    url="https://www.sbert.net/docs/package_reference/sentence_transformer/evaluation.html",
    notes="Reference for MRR, nDCG, MAP, precision@k and recall@k retrieval metrics.",
)
RAGAS = ReferenceSource(
    name="Ragas metrics",
    kind="external_framework",
    url="https://docs.ragas.io/en/stable/concepts/metrics/",
    notes="Reference for context precision, context recall, faithfulness and response relevancy.",
)
RAGAS_NOISE = ReferenceSource(
    name="Ragas NoiseSensitivity",
    kind="external_framework",
    url="https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/noise_sensitivity/",
    notes="Reference for distractor/noise sensitivity over retrieved contexts.",
)
DEEPEVAL_TOOL = ReferenceSource(
    name="DeepEval ToolCorrectnessMetric",
    kind="external_framework",
    url="https://deepeval.com/docs/metrics-tool-correctness",
    notes="Reference for tool-use correctness with expected tool calls.",
)
DEEPEVAL_ARGUMENT = ReferenceSource(
    name="DeepEval ArgumentCorrectnessMetric",
    kind="external_framework",
    url="https://deepeval.com/docs/metrics-argument-correctness",
    notes="Reference for tool argument correctness.",
)
DEEPEVAL_TASK = ReferenceSource(
    name="DeepEval TaskCompletionMetric",
    kind="external_framework",
    url="https://deepeval.com/docs/metrics-task-completion",
    notes="Reference for trace-level task completion.",
)
DEEPEVAL_STEP = ReferenceSource(
    name="DeepEval StepEfficiencyMetric",
    kind="external_framework",
    url="https://deepeval.com/docs/metrics-step-efficiency",
    notes="Reference for trace-level execution efficiency and redundant steps.",
)
PROMPTFOO = ReferenceSource(
    name="promptfoo assertions",
    kind="external_framework",
    url="https://www.promptfoo.dev/docs/configuration/expected-outputs/",
    notes="Reference for regression assertions, JSON checks, rubric and trajectory grading.",
)
EVIDENTLY = ReferenceSource(
    name="Evidently descriptors",
    kind="external_framework",
    url="https://docs.evidentlyai.com/metrics/all_descriptors",
    notes="Reference for deterministic text descriptors, JSON checks and LLM text evals.",
)
RULER = ReferenceSource(
    name="NVIDIA RULER",
    kind="external_framework",
    url="https://github.com/NVIDIA/RULER",
    notes="Reference benchmark pattern for configurable long-context length and position tests.",
)
HELMET = ReferenceSource(
    name="Princeton HELMET",
    kind="external_framework",
    url="https://github.com/princeton-nlp/HELMET",
    notes="Reference benchmark pattern for application-centric long-context evaluations.",
)
PHOENIX = ReferenceSource(
    name="Phoenix RAG evaluation",
    kind="external_framework",
    url="https://arize.com/docs/phoenix/cookbook/evaluation/evaluate-rag",
    notes="Reference workflow for separating retrieval and response evaluation.",
)
GOLDEN_FIXTURES = ReferenceSource(
    name="OVIQS deterministic golden fixtures",
    kind="deterministic_golden_fixture",
    notes="Small fixed inputs/logits/traces with explicit expected pass/fail behavior.",
)


_REFERENCES: tuple[MetricReference, ...] = (
    MetricReference(
        family="likelihood",
        metric_names=(
            "token_logprobs",
            "mean_log_prob",
            "num_tokens",
            "nll",
            "nll_delta_vs_ref",
            "perplexity",
            "ppl_relative_delta_vs_ref",
            "sliding_window_ppl",
            "word_perplexity",
            "byte_perplexity",
            "bits_per_byte",
            "length_bucketed_ppl",
            "effective_context_bucketed_ppl",
            "position_bucketed_ppl",
        ),
        primary_reference="lm-evaluation-harness loglikelihood/loglikelihood_rolling",
        sources=(LM_EVAL, LIGHTEVAL, HF_PPL, GOLDEN_FIXTURES),
        oracle=(
            "Compare token-aligned NLL/PPL against lm-eval/LightEval semantics or "
            "deterministic logits fixtures."
        ),
        degradation_rule=(
            "Degraded when current NLL/PPL or configured delta exceeds the reference baseline gate."
        ),
        required_inputs=("aligned logits", "input_ids", "attention_mask or explicit score mask"),
    ),
    MetricReference(
        family="distribution_drift",
        metric_names=(
            "kl_per_pos",
            "mean_kl",
            "p95_kl",
            "max_kl",
            "js_per_pos",
            "mean_js",
            "p95_js",
            "ref_entropy_per_pos",
            "cur_entropy_per_pos",
            "entropy_drift_per_pos",
            "mean_entropy_drift",
            "logit_cosine_per_pos",
            "mean_logit_cosine",
            "top1_changed_rate",
            "top5_overlap",
            "top10_overlap",
            "topk_overlap",
            "target_rank_delta",
            "sensitive_token_drift",
        ),
        primary_reference="SciPy entropy/KL formulas plus deterministic logits fixtures",
        sources=(SCIPY_ENTROPY, GOLDEN_FIXTURES),
        oracle=(
            "Compare same-position full-vocabulary distributions; identical logits must "
            "have zero KL/JS and cosine 1."
        ),
        degradation_rule="Degraded when drift exceeds the configured KL/JS/top-k/cosine gate.",
        required_inputs=("reference logits", "current logits", "identical tokenizer and positions"),
    ),
    MetricReference(
        family="long_context",
        metric_names=(
            "context_gain",
            "context_gain_64k",
            "nll_by_context_length",
            "context_saturation_curve",
            "lost_in_middle_score",
            "degradation_slope",
            "degradation_slope_quality",
            "distractor_sensitivity",
            "clean_nll",
            "distracted_nll",
            "faithfulness_drop",
            "supported_claim_ratio_drop",
            "context_gain_drop",
            "entropy_shift_with_distractors",
            "authoritative_margin",
            "candidate_logprobs",
            "conflict_nll",
            "conflict_sensitivity",
            "conflict_entropy",
            "source_mixup_rate",
            "unsupported_resolution_rate",
            "conflict_contradiction_rate",
        ),
        primary_reference="RULER/HELMET-style controlled length, position and distractor suites",
        sources=(RULER, HELMET, RAGAS_NOISE, GOLDEN_FIXTURES),
        oracle=(
            "Compare quality/NLL at controlled lengths, positions, distractor variants "
            "or conflict labels."
        ),
        degradation_rule=(
            "Degraded when longer/noisier/conflicting context crosses its configured "
            "quality-drop gate."
        ),
        required_inputs=(
            "controlled long-context samples",
            "baseline result",
            "current length or perturbation result",
        ),
    ),
    MetricReference(
        family="generation",
        metric_names=(
            "ngram_repetition_rate",
            "ngram_repetition",
            "repetition_rate",
            "unique_ngram_ratio",
            "duplicate_sentence_ratio",
            "topic_drift",
            "entity_preservation_rate",
            "entity_hallucination_rate",
            "entity_contradiction_rate",
            "date_number_version_mismatch_rate",
            "json_validity",
            "json_valid",
            "schema_validity",
            "required_section_coverage",
            "forbidden_section_violation",
            "markdown_structure_score",
        ),
        primary_reference="Evidently/promptfoo descriptors and assertions",
        sources=(EVIDENTLY, PROMPTFOO, GOLDEN_FIXTURES),
        oracle=(
            "Compare output against deterministic structure/entity checks or an "
            "explicit judge rubric."
        ),
        degradation_rule=(
            "Degraded when deterministic checks fail or judge/semantic score drops below the gate."
        ),
        required_inputs=("output text", "expected structure/entities/rubric"),
    ),
    MetricReference(
        family="serving",
        metric_names=(
            "batch_invariance",
            "batch_invariance_drift",
            "batch_invariance_mean_kl",
            "batch_mean_kl",
            "batch_p95_kl",
            "batch_js",
            "batch_entropy_drift",
            "batch_top1_changed_rate",
            "batch_generation_prefix_divergence",
            "generation_prefix_divergence",
            "prefix_divergence_rate",
            "kv_cache_drift",
            "kv_cache_mean_kl",
            "kv_cache_p95_kl",
            "kv_mean_kl",
            "kv_p95_kl",
            "kv_mean_js",
            "kv_entropy_drift",
            "kv_top1_change_rate",
            "kv_generation_divergence",
            "device_drift",
        ),
        primary_reference="same-sample deterministic serving invariance oracle",
        sources=(SCIPY_ENTROPY, PROMPTFOO, GOLDEN_FIXTURES),
        oracle=(
            "Compare the same prompt alone vs batch, full-forward vs KV-cache, or "
            "device A vs device B."
        ),
        degradation_rule=(
            "Degraded when serving variant drift or prefix divergence exceeds its gate."
        ),
        required_inputs=("baseline serving output/logits", "variant serving output/logits"),
    ),
    MetricReference(
        family="rag",
        metric_names=(
            "recall_at_k",
            "precision_at_k",
            "mrr",
            "ndcg",
            "rank_quality",
            "evidence_coverage",
            "token_waste_ratio",
            "distractor_ratio",
            "context_precision",
            "context_recall",
            "faithfulness",
            "supported_claim_ratio",
            "unsupported_claim_rate",
            "contradiction_rate",
            "answer_relevance",
            "answer_relevancy",
            "answer_relevance_lexical",
            "citation_precision",
            "citation_recall",
            "source_correctness",
            "faithfulness_rule_based",
        ),
        primary_reference="Ragas plus IR metrics from SentenceTransformers/sklearn",
        sources=(RAGAS, SENTENCE_TRANSFORMERS_IR, SCIKIT_NDCG, PHOENIX, GOLDEN_FIXTURES),
        oracle=(
            "Compare retrieval labels, reference contexts, claims, answers and citations "
            "against explicit ground truth."
        ),
        degradation_rule=(
            "Degraded when retrieval, grounding, answer or citation score falls below its gate."
        ),
        required_inputs=(
            "question",
            "retrieved contexts",
            "reference answer/evidence or judge labels",
        ),
    ),
    MetricReference(
        family="agent",
        metric_names=(
            "tool_call_validity",
            "tool_correctness",
            "argument_correctness",
            "redundant_tool_call_rate",
            "step_efficiency",
            "agent_state_drift",
            "state_drift_score",
            "observation_grounding_score",
            "task_completion",
            "policy_violation_rate",
            "recovery_score",
            "recovery_after_tool_error",
            "same_error_repeat_rate",
            "fallback_quality_score",
            "unsafe_recovery_rate",
            "unnecessary_user_clarification_rate",
        ),
        primary_reference="DeepEval agent metrics plus deterministic trace fixtures",
        sources=(DEEPEVAL_TOOL, DEEPEVAL_ARGUMENT, DEEPEVAL_TASK, DEEPEVAL_STEP, GOLDEN_FIXTURES),
        oracle=(
            "Compare trace steps, expected tools/state, observations and recovery "
            "scenarios against labels or fixtures."
        ),
        degradation_rule=(
            "Degraded when tool/state/grounding/task/recovery score violates its configured gate."
        ),
        required_inputs=(
            "agent trace",
            "tool schemas",
            "expected tools/state/outcome or judge labels",
        ),
    ),
    MetricReference(
        family="performance",
        metric_names=(
            "forward_latency_ms_mean",
            "forward_latency_ms_p95",
            "tokens_per_second_forward",
            "generation_latency_ms",
        ),
        primary_reference="same-hardware performance baseline",
        sources=(GOLDEN_FIXTURES,),
        oracle=(
            "Compare latency and throughput against a same-model, same-device baseline "
            "captured under documented run conditions."
        ),
        degradation_rule=(
            "Degraded when latency increases or throughput drops beyond the configured "
            "same-hardware gate."
        ),
        required_inputs=("model", "device", "dataset or prompt", "baseline performance report"),
    ),
)


_BY_METRIC: dict[str, MetricReference] = {
    metric_name: reference for reference in _REFERENCES for metric_name in reference.metric_names
}


def list_metric_references() -> list[MetricReference]:
    return list(_REFERENCES)


def references_for_family(family: str) -> list[MetricReference]:
    return [reference for reference in _REFERENCES if reference.family == family]


def get_metric_reference(metric_name: str) -> MetricReference | None:
    return _BY_METRIC.get(metric_name)


def require_metric_reference(metric_name: str) -> MetricReference:
    reference = get_metric_reference(metric_name)
    if reference is None:
        raise KeyError(f"No OVIQS metric reference/oracle registered for metric: {metric_name}")
    return reference


def build_report_reference_manifest(
    report: dict[str, Any],
    *,
    include_unknown_warnings: bool = True,
) -> dict[str, Any]:
    """Build reference metadata for scalar metrics present in a report payload."""

    manifest: dict[str, Any] = {}
    missing: dict[str, list[str]] = {}
    for section in _REPORT_SECTIONS:
        payload = report.get(section)
        if not isinstance(payload, dict):
            continue
        section_refs: dict[str, Any] = {}
        for metric_path, metric_name, _value in _iter_scalar_metric_items(payload):
            reference = get_metric_reference(metric_name)
            if reference is None:
                missing.setdefault(section, []).append(metric_path)
                continue
            section_refs[metric_path] = reference.to_dict()
        if section_refs:
            manifest[section] = section_refs
    if include_unknown_warnings and missing:
        manifest["_warnings"] = [
            {
                "section": section,
                "metrics_without_reference": sorted(metric_names),
            }
            for section, metric_names in sorted(missing.items())
        ]
    return manifest


def _iter_scalar_metric_items(
    payload: dict[str, Any],
    *,
    prefix: str = "",
) -> list[tuple[str, str, Any]]:
    items: list[tuple[str, str, Any]] = []
    for key, value in payload.items():
        if key in _NON_METRIC_KEYS:
            continue
        metric_path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            if get_metric_reference(key) is not None:
                items.append((metric_path, key, None))
                continue
            items.extend(_iter_scalar_metric_items(value, prefix=metric_path))
            continue
        if isinstance(value, list):
            continue
        if _is_scalar_metric_value(value):
            items.append((metric_path, key, value))
    return items


def _is_scalar_metric_value(value: Any) -> bool:
    return isinstance(value, int | float | str | bool) or value is None


_NON_METRIC_KEYS = {
    "id",
    "status",
    "warnings",
    "warning",
    "error",
    "reason",
    "sample_output",
    "samples",
    "raw",
    "n",
    "total_ngrams",
    "prefix_match_tokens",
    "alone_tokens",
    "batched_tokens",
    "tool_calls",
    "valid_tool_calls",
    "redundant_tool_calls",
    "checked_steps",
    "policy_violations",
    "tool_errors",
    "recovered_tool_errors",
    "expected_evidence",
    "matched_evidence",
    "retrieved_contexts",
    "relevant_contexts",
    "distractor_contexts",
    "expected_citations",
    "actual_citations",
    "matched_citations",
    "claims",
    "supported_claims",
    "input_tokens",
    "generated_tokens_whitespace_estimate",
    "task_completed",
    "state_errors",
    "errors",
    "dataset",
    "model_context_limit_tokens",
    "serving_max_tokens",
}

_REPORT_SECTIONS = (
    "inference_equivalence",
    "likelihood",
    "long_context",
    "robustness",
    "generation",
    "rag",
    "agent",
    "serving",
    "performance",
    "likelihood_wikitext2",
    "cpu_gpu_drift",
    "precision_drift_fp16_vs_int8",
    "long_context_controlled",
    "distractor_conflict",
    "serving_extended",
    "generation_extended",
    "rag_extended",
    "agent_extended",
)
