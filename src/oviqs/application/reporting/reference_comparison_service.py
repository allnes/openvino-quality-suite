from __future__ import annotations

import json
from typing import Any

from oviqs.domain.references import get_metric_reference

DEFAULT_REFERENCE_COMPARISON_METRICS = (
    ("likelihood", "nll"),
    ("likelihood", "perplexity"),
    ("likelihood", "bits_per_byte"),
    ("inference_equivalence", "mean_kl"),
    ("inference_equivalence", "p95_kl"),
    ("inference_equivalence", "top1_changed_rate"),
    ("generation", "json_validity"),
    ("generation", "schema_validity"),
    ("generation", "repetition_rate"),
    ("long_context", "lost_in_middle_score"),
    ("long_context", "distractor_sensitivity"),
    ("long_context", "authoritative_margin"),
    ("long_context", "contradiction_rate"),
    ("serving", "batch_mean_kl"),
    ("serving", "batch_p95_kl"),
    ("serving", "batch_top1_changed_rate"),
    ("serving", "kv_mean_kl"),
    ("serving", "kv_top1_change_rate"),
    ("rag", "context_precision"),
    ("rag", "context_recall"),
    ("rag", "faithfulness"),
    ("agent", "tool_call_validity"),
    ("agent", "recovery_score"),
    ("performance", "forward_latency_ms_mean"),
    ("performance", "generation_latency_ms"),
)


class ReferenceComparisonService:
    def build(
        self,
        reports: list[tuple[str, dict[str, Any], str]],
        metrics: list[tuple[str, str]] | None = None,
        *,
        include_all_metrics: bool = False,
    ) -> dict[str, Any]:
        metric_specs = (
            collect_metric_coverage_specs([report for _label, report, _path in reports])
            if include_all_metrics
            else metrics or list(DEFAULT_REFERENCE_COMPARISON_METRICS)
        )
        rows = []
        for section, metric in metric_specs:
            row: dict[str, Any] = {
                "section": section,
                "metric": metric,
                **_reference_columns(reports, section, metric),
            }
            for label, report, _path in reports:
                value_path, value = find_metric_value(report.get(section, {}), metric)
                row[label] = format_metric_value(value)
                row[f"{label}_status"] = metric_status(report, section, metric, value)
                row[f"{label}_path"] = f"{section}.{value_path}" if value_path else ""
            rows.append(row)
        return {
            "reports": [{"label": label, "path": path} for label, _report, path in reports],
            "rows": rows,
        }


def collect_metric_coverage_specs(reports: list[dict[str, Any]]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    seen = set()
    for report in reports:
        entries = report.get("metric_coverage", {}).get("entries", [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            section = entry.get("section")
            metric = entry.get("metric")
            if not isinstance(section, str) or not isinstance(metric, str):
                continue
            key = (section, metric)
            if key in seen:
                continue
            seen.add(key)
            specs.append(key)
    if specs:
        return specs
    return list(DEFAULT_REFERENCE_COMPARISON_METRICS)


def find_metric_reference(
    report: dict[str, Any],
    section: str,
    metric: str,
) -> dict[str, Any] | None:
    refs = report.get("metric_references", {})
    if not isinstance(refs, dict):
        return None
    section_refs = refs.get(section, {})
    if not isinstance(section_refs, dict):
        return None
    for key, value in section_refs.items():
        if not isinstance(value, dict):
            continue
        metric_names = value.get("metric_names", [])
        if key == metric or key.endswith(f".{metric}") or metric in metric_names:
            return value
    return None


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


def metric_status(
    report: dict[str, Any],
    section: str,
    metric: str,
    value: Any,
) -> str:
    if value is None:
        reason = _coverage_reason(report, section, metric)
        return f"unknown ({reason})" if reason else "unknown"
    section_payload = report.get(section)
    section_status = section_payload.get("status") if isinstance(section_payload, dict) else None
    if section_status and section_status not in {"pass", "measured"}:
        return f"section_{section_status}"
    return "measured"


def format_metric_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _reference_columns(
    reports: list[tuple[str, dict[str, Any], str]],
    section: str,
    metric: str,
) -> dict[str, Any]:
    for _label, report, _path in reports:
        reference = find_metric_reference(report, section, metric)
        if reference:
            return {
                "reference": str(reference.get("primary_reference", "")),
                "degradation_rule": str(reference.get("degradation_rule", "")),
                "oracle": str(reference.get("oracle", "")),
                "sources": reference.get("sources", []),
            }
    reference_obj = get_metric_reference(metric)
    if reference_obj is None:
        return {"reference": "", "degradation_rule": ""}
    return {
        "reference": reference_obj.primary_reference,
        "degradation_rule": reference_obj.degradation_rule,
        "oracle": reference_obj.oracle,
        "sources": [source.to_dict() for source in reference_obj.sources],
    }


def _coverage_reason(report: dict[str, Any], section: str, metric: str) -> str:
    coverage = report.get("metric_coverage", {}).get("entries", [])
    if not isinstance(coverage, list):
        return ""
    for item in coverage:
        if item.get("section") == section and item.get("metric") == metric:
            return str(item.get("reason") or "")
    return ""


__all__ = [
    "DEFAULT_REFERENCE_COMPARISON_METRICS",
    "ReferenceComparisonService",
    "collect_metric_coverage_specs",
    "find_metric_reference",
    "find_metric_value",
    "format_metric_value",
    "metric_status",
]
