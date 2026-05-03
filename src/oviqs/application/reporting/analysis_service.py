from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, cast

from oviqs.application.reporting.comparison_service import ReportComparisonService
from oviqs.domain.reporting import AnalysisFinding, MetricObservation
from oviqs.domain.reporting.analysis import AnalysisSummary, ReportAnalysis
from oviqs.domain.reporting.severity import ReportStatus, Severity, worst_status
from oviqs.ports.analysis import AnalysisRulePort, MetricCatalogPort, TrendStorePort


class ReportAnalysisService:
    def __init__(
        self,
        comparison_service: ReportComparisonService | None = None,
        *,
        rules: Sequence[AnalysisRulePort] = (),
        metric_catalog: MetricCatalogPort | None = None,
        trend_store: TrendStorePort | None = None,
    ) -> None:
        self.comparison_service = comparison_service or ReportComparisonService(trend_store)
        self.rules = tuple(rules)
        self.metric_catalog = metric_catalog

    def analyze(
        self,
        report: dict[str, Any],
        *,
        baseline: dict[str, Any] | None = None,
        gates: dict[str, Any] | None = None,
    ) -> ReportAnalysis:
        comparison = self.comparison_service.compare(report, baseline=baseline, gates=gates)
        findings = self._findings(report, comparison)
        counts = Counter(metric.status for metric in comparison.metrics)
        statuses: list[str] = [str(report.get("summary", {}).get("overall_status", "unknown"))]
        statuses.extend(metric.status for metric in comparison.metrics)
        summary = AnalysisSummary(
            overall_status=worst_status(statuses),
            passed=counts["pass"],
            warning=counts["warning"],
            failed=counts["fail"],
            unknown=counts["unknown"],
            finding_count=len(findings),
        )
        return ReportAnalysis(
            summary=summary,
            findings=tuple(findings),
            metrics=comparison.metrics,
            biggest_regressions=comparison.biggest_regressions,
            biggest_improvements=comparison.biggest_improvements,
            unknown_metrics=comparison.unknown_metrics,
            sample_outliers=tuple(_sample_outliers(report)),
            trend_points=comparison.trend_points,
        )

    def _findings(self, report: dict[str, Any], comparison: Any) -> list[AnalysisFinding]:
        findings: list[AnalysisFinding] = []
        for section, payload in report.items():
            if not isinstance(payload, dict):
                continue
            status = payload.get("status")
            if status not in {"warning", "fail", "unknown"}:
                continue
            findings.append(
                self._section_finding(
                    section,
                    status,
                    [metric.path for metric in comparison.metrics if metric.section == section][:5],
                    payload.get("warnings"),
                )
            )
        unknown = [metric for metric in comparison.unknown_metrics if metric.value is None]
        if unknown:
            findings.append(
                AnalysisFinding(
                    id="unknown-metrics",
                    title=f"{len(unknown)} metrics are unknown",
                    severity="medium",
                    category="coverage",
                    status="unknown",
                    evidence_paths=tuple(metric.path for metric in unknown[:10]),
                    impact="Quality gates cannot prove regression status for missing metrics.",
                    recommendation=(
                        "Expose the required evidence, register a metric reference, or remove "
                        "the gate."
                    ),
                    details={"count": len(unknown)},
                )
            )
        for metric in comparison.biggest_regressions[:5]:
            findings.append(
                AnalysisFinding(
                    id=f"regression-{metric.path.replace('.', '-')}",
                    title=f"{metric.path} regressed",
                    severity="high" if metric.status == "fail" else "medium",
                    category=metric.section,
                    status=_finding_status(metric.status),
                    evidence_paths=(metric.path,),
                    impact="Current metric is worse than the supplied baseline.",
                    recommendation=metric.degradation_rule
                    or "Inspect model export, precision, tokenizer alignment and runtime path.",
                    details={
                        "current": metric.value,
                        "baseline": metric.baseline_value,
                        "delta_abs": metric.delta_abs,
                        "delta_rel": metric.delta_rel,
                    },
                )
            )
        findings.extend(self._reference_gap_findings(comparison.metrics))
        for rule in self.rules:
            findings.extend(rule.analyze(report, list(comparison.metrics)))
        return sorted(findings, key=lambda item: (-_severity_rank(item.severity), item.id))

    def _reference_gap_findings(
        self,
        metrics: tuple[MetricObservation, ...],
    ) -> list[AnalysisFinding]:
        if self.metric_catalog is None:
            return []
        missing = [
            metric
            for metric in metrics
            if "gated" in metric.tags and self.metric_catalog.get_reference(metric.name) is None
        ]
        if not missing:
            return []
        return [
            AnalysisFinding(
                id="missing-metric-references",
                title=f"{len(missing)} gated metrics have no registered reference",
                severity="medium",
                category="coverage",
                status="unknown",
                evidence_paths=tuple(metric.path for metric in missing[:10]),
                impact="Quality gates cannot prove whether these metrics degraded.",
                recommendation="Register metric references/oracles or disable the gates.",
                details={"count": len(missing)},
            )
        ]

    def _section_finding(
        self,
        section: str,
        status: str,
        evidence_paths: list[str],
        warnings: Any,
    ) -> AnalysisFinding:
        severity: Severity = (
            "high" if status == "fail" else ("medium" if status == "warning" else "low")
        )
        recommendation = _recommendation_for_section(section)
        details: dict[str, Any] = {}
        if isinstance(warnings, list):
            details["warnings"] = warnings
        return AnalysisFinding(
            id=f"section-{section}-{status}",
            title=f"{section.replace('_', ' ').title()} section is {status}",
            severity=severity,
            category=section,
            status=_finding_status(status),
            evidence_paths=tuple(evidence_paths),
            impact="The section status affects the top-level report quality signal.",
            recommendation=recommendation,
            details=details,
        )


def _finding_status(status: str) -> ReportStatus:
    if status in {"warning", "fail", "unknown"}:
        return cast(ReportStatus, status)
    return "warning"


def _severity_rank(severity: str) -> int:
    return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(severity, 0)


def _recommendation_for_section(section: str) -> str:
    recommendations = {
        "likelihood": "Inspect tokenizer alignment, precision, model export and calibration data.",
        "inference_equivalence": "Compare aligned logits and confirm CPU/GPU inputs are identical.",
        "long_context": (
            "Inspect KV-cache, context packing, RoPE scaling and context limit handling."
        ),
        "serving": "Compare single/batch paths and reset cache state between requests.",
        "rag": "Inspect retrieval candidates, citations and answer evidence mapping.",
        "agent": "Inspect tool-call schema, trace events and recovery after errors.",
        "generation": (
            "Inspect prompt contract, structured output constraints and deterministic decoding."
        ),
        "performance": "Inspect compile cache, device selection and batch/sequence shape.",
    }
    return recommendations.get(section, "Inspect section evidence and configured gates.")


def _sample_outliers(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section, payload in report.items():
        if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
            continue
        samples = [sample for sample in payload["samples"] if isinstance(sample, dict)]
        for metric_name in _numeric_sample_metric_names(samples):
            values = [
                float(sample[metric_name])
                for sample in samples
                if _is_number(sample.get(metric_name))
            ]
            if len(values) < 3:
                continue
            center = sum(values) / len(values)
            ranked = sorted(
                (
                    (
                        abs(float(sample[metric_name]) - center),
                        idx,
                        sample,
                    )
                    for idx, sample in enumerate(samples)
                    if _is_number(sample.get(metric_name))
                ),
                reverse=True,
                key=lambda item: item[0],
            )
            for distance, idx, sample in ranked[:3]:
                if distance == 0:
                    continue
                rows.append(
                    {
                        "section": section,
                        "sample_index": idx,
                        "sample_id": sample.get("id"),
                        "metric": metric_name,
                        "value": sample.get(metric_name),
                        "mean": center,
                        "distance": distance,
                    }
                )
    return sorted(rows, key=lambda item: item["distance"], reverse=True)[:20]


def _numeric_sample_metric_names(samples: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for sample in samples:
        for key, value in sample.items():
            if key not in {"id", "sample_id"} and _is_number(value):
                names.add(key)
    return sorted(names)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


__all__ = ["ReportAnalysisService"]
