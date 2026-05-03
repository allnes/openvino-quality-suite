from __future__ import annotations

from typing import Any

from oviqs.domain.reporting import AnalysisFinding, MetricObservation
from oviqs.domain.reporting.severity import ReportStatus, Severity


class BuiltInAnalysisRules:
    def analyze(
        self,
        report: dict[str, Any],
        metrics: list[MetricObservation],
    ) -> list[AnalysisFinding]:
        by_path = {metric.path: metric for metric in metrics}
        findings: list[AnalysisFinding] = []
        findings.extend(_entropy_drift_findings(by_path))
        findings.extend(_long_context_findings(by_path))
        findings.extend(_serving_findings(by_path))
        findings.extend(_rag_findings(by_path))
        findings.extend(_agent_findings(by_path))
        return findings


def _entropy_drift_findings(metrics: dict[str, MetricObservation]) -> list[AnalysisFinding]:
    return _threshold_findings(
        metrics,
        names=("mean_entropy_drift", "entropy_drift", "entropy_shift_with_distractors"),
        finding_id="entropy-drift",
        title="Entropy drift needs review",
        category="inference_equivalence",
        impact="Distribution confidence changed enough to affect token ranking and generation.",
        recommendation="Inspect quantization precision, logits alignment and tokenizer parity.",
    )


def _long_context_findings(metrics: dict[str, MetricObservation]) -> list[AnalysisFinding]:
    return _threshold_findings(
        metrics,
        names=("lost_in_middle_score", "degradation_slope", "context_saturation_curve"),
        finding_id="long-context-saturation",
        title="Long-context behavior needs review",
        category="long_context",
        impact=(
            "Context position or length sensitivity may hide retrieval or prompt-packing "
            "regressions."
        ),
        recommendation="Inspect KV-cache, RoPE scaling, prompt packing and context export path.",
    )


def _serving_findings(metrics: dict[str, MetricObservation]) -> list[AnalysisFinding]:
    return _threshold_findings(
        metrics,
        names=("batch_mean_kl", "batch_p95_kl", "batch_js", "batch_top1_changed_rate"),
        finding_id="serving-batch-invariance",
        title="Serving batch invariance needs review",
        category="serving",
        impact=(
            "Single-request and batched serving paths may produce different token distributions."
        ),
        recommendation=(
            "Compare single/batch paths, padding, beam index handling and cache state reset."
        ),
    )


def _rag_findings(metrics: dict[str, MetricObservation]) -> list[AnalysisFinding]:
    return _threshold_findings(
        metrics,
        names=("citation_precision", "citation_recall", "faithfulness", "supported_claim_ratio"),
        finding_id="rag-citation-grounding",
        title="RAG citation or grounding quality needs review",
        category="rag",
        impact="Answers may be unsupported or mapped to the wrong retrieved evidence.",
        recommendation="Inspect retrieval candidates, citation ids and answer evidence mapping.",
    )


def _agent_findings(metrics: dict[str, MetricObservation]) -> list[AnalysisFinding]:
    return _threshold_findings(
        metrics,
        names=("recovery_score", "task_completion", "tool_call_validity", "same_error_repeat_rate"),
        finding_id="agent-recovery-tool-use",
        title="Agent recovery or tool-use quality needs review",
        category="agent",
        impact=(
            "The agent may repeat failed actions, miss required tools or finish without grounding."
        ),
        recommendation=(
            "Inspect trace events, tool schema validation, state drift and recovery paths."
        ),
    )


def _threshold_findings(
    metrics: dict[str, MetricObservation],
    *,
    names: tuple[str, ...],
    finding_id: str,
    title: str,
    category: str,
    impact: str,
    recommendation: str,
) -> list[AnalysisFinding]:
    evidence = [
        metric
        for metric in metrics.values()
        if metric.name in names and metric.status in {"warning", "fail", "unknown"}
    ]
    if not evidence:
        return []
    status: ReportStatus = (
        "fail" if any(metric.status == "fail" for metric in evidence) else "warning"
    )
    if all(metric.status == "unknown" for metric in evidence):
        status = "unknown"
    severity: Severity = "high" if status == "fail" else "medium"
    return [
        AnalysisFinding(
            id=finding_id,
            title=title,
            severity=severity,
            category=category,
            status=status,
            evidence_paths=tuple(metric.path for metric in evidence[:8]),
            impact=impact,
            recommendation=recommendation,
            details={
                "metrics": [
                    {
                        "path": metric.path,
                        "value": metric.value,
                        "threshold": metric.threshold,
                        "status": metric.status,
                    }
                    for metric in evidence[:8]
                ]
            },
        )
    ]


__all__ = ["BuiltInAnalysisRules"]
