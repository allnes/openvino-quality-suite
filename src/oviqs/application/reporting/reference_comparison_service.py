from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from oviqs.application.reporting.normalization import REPORT_SECTIONS
from oviqs.domain.references import MetricReference, get_metric_reference, list_metric_references

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
                row[f"{label}_reason"] = metric_reason(report, section, metric, value, value_path)
                row[f"{label}_path"] = f"{section}.{value_path}" if value_path else ""
            rows.append(row)
        return {
            "reports": [{"label": label, "path": path} for label, _report, path in reports],
            "rows": rows,
        }


def collect_metric_coverage_specs(reports: list[dict[str, Any]]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    seen = set()

    def add(section: str, metric: str) -> None:
        key = (section, metric)
        if key in seen:
            return
        seen.add(key)
        specs.append(key)

    for report in reports:
        for section, metric in supported_metric_specs():
            add(section, metric)
        entries = report.get("metric_coverage", {}).get("entries", [])
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                entry_section = entry.get("section")
                entry_metric = entry.get("metric")
                if not isinstance(entry_section, str) or not isinstance(entry_metric, str):
                    continue
                add(entry_section, entry_metric)
        for section, metric_path in discover_report_metric_specs(report):
            add(section, metric_path)
    if specs:
        return specs
    return list(DEFAULT_REFERENCE_COMPARISON_METRICS)


def supported_metric_specs() -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    seen = set()
    for reference in list_metric_references():
        section = _section_for_reference_family(reference.family)
        for metric in reference.metric_names:
            key = (section, metric)
            if key in seen:
                continue
            seen.add(key)
            specs.append(key)
    return specs


def discover_report_metric_specs(report: dict[str, Any]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for section in REPORT_SECTIONS:
        payload = report.get(section)
        if not isinstance(payload, dict):
            continue
        for path, _value in _walk_metric_scalars(payload):
            specs.append((section, path))
    return specs


def _walk_metric_scalars(
    payload: dict[str, Any],
    prefix: tuple[str, ...] = (),
) -> Iterator[tuple[str, Any]]:
    for key, value in payload.items():
        key = str(key)
        if key in _NON_METRIC_KEYS:
            continue
        path = (*prefix, key)
        if isinstance(value, int | float | str | bool | None):
            yield ".".join(path), value
        elif isinstance(value, dict):
            yield from _walk_metric_scalars(value, path)
        elif isinstance(value, list):
            if not _is_scalar_metric_list(value):
                continue
            yield ".".join(path), value


def _is_scalar_metric_list(value: list[Any]) -> bool:
    return bool(value) and all(isinstance(item, int | float | str | bool | None) for item in value)


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
    candidate_names = _metric_reference_candidates(metric)
    for key, value in section_refs.items():
        if not isinstance(value, dict):
            continue
        metric_names = value.get("metric_names", [])
        if (
            key in candidate_names
            or any(key.endswith(f".{candidate}") for candidate in candidate_names)
            or any(candidate in metric_names for candidate in candidate_names)
        ):
            return value
    return None


def find_metric_value(payload: Any, metric: str, prefix: str = "") -> tuple[str | None, Any]:
    if not isinstance(payload, dict):
        return None, None
    derived = _derived_metric_value(payload, metric)
    if derived[0] is not None:
        return derived
    if "." in metric:
        value = _lookup_dotted_metric(payload, metric)
        if value[0] is not None:
            return value
    if metric in payload:
        return metric, payload[metric]
    for alias in _metric_alias_paths(metric):
        value = _lookup_dotted_metric(payload, alias)
        if value[0] is not None:
            return value
    leaf = metric.rsplit(".", 1)[-1]
    if leaf in payload:
        return leaf, payload[leaf]
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            found_path, found_value = find_metric_value(value, metric, path)
            if found_path:
                return found_path, found_value
    return None, None


def _derived_metric_value(payload: dict[str, Any], metric: str) -> tuple[str | None, Any]:
    if metric == "mean_log_prob":
        path, value = _lookup_dotted_metric(payload, "nll")
        if path is not None and isinstance(value, int | float) and not isinstance(value, bool):
            return path, -float(value)
    return None, None


def _metric_alias_paths(metric: str) -> tuple[str, ...]:
    return _METRIC_VALUE_ALIASES.get(metric, ())


def metric_status(
    report: dict[str, Any],
    section: str,
    metric: str,
    value: Any,
) -> str:
    if value is None:
        coverage_status = _coverage_status(report, section, metric)
        if coverage_status:
            return coverage_status
        section_payload = report.get(section)
        section_status = (
            section_payload.get("status") if isinstance(section_payload, dict) else None
        )
        if section_status == "fail":
            return "failed"
        if section_status == "unknown":
            return "unknown"
        return _missing_metric_status(metric)
    section_payload = report.get(section)
    section_status = section_payload.get("status") if isinstance(section_payload, dict) else None
    if (
        isinstance(value, bool)
        and value is False
        and metric.rsplit(".", 1)[-1]
        in {
            "json_valid",
            "schema_valid",
        }
    ):
        return "fail"
    value_status = _value_metric_status(metric, value)
    if value_status:
        return value_status
    if section_status in {"warning", "fail", "unknown"}:
        return section_status
    return "measured"


def metric_reason(
    report: dict[str, Any],
    section: str,
    metric: str,
    value: Any,
    value_path: str | None,
) -> str:
    if value is not None:
        value_reason = _value_metric_reason(metric, value)
        if value_reason:
            return value_reason
        section_payload = report.get(section)
        section_status = (
            section_payload.get("status") if isinstance(section_payload, dict) else None
        )
        gate_note = _gate_note(report, section, metric)
        if gate_note:
            return gate_note
        if section_status in {"warning", "fail", "unknown"}:
            section_reason = _section_reason(section_payload)
            if section_reason:
                return section_reason
        path = f"{section}.{value_path}" if value_path else section
        return f"measured from {path}; no metric-level gate was found in this report"

    coverage_reason = _coverage_reason(report, section, metric)
    if coverage_reason:
        return coverage_reason

    section_payload = report.get(section)
    if isinstance(section_payload, dict):
        if section_payload.get("status") == "fail":
            return (
                _section_reason(section_payload) or "section failed before this metric was emitted"
            )
        if section_payload.get("status") == "unknown":
            section_reason = _section_reason(section_payload)
            if section_reason:
                return section_reason

    reference = _reference_for_metric(metric)
    if reference is not None and reference.required_inputs:
        return "requires " + ", ".join(reference.required_inputs)
    return "metric is supported by the catalog but was not emitted by this report"


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
                "required_inputs": reference.get("required_inputs", []),
                "sources": reference.get("sources", []),
            }
    reference_obj = _reference_for_metric(metric)
    if reference_obj is None:
        return {"reference": "", "degradation_rule": "", "required_inputs": []}
    return {
        "reference": reference_obj.primary_reference,
        "degradation_rule": reference_obj.degradation_rule,
        "oracle": reference_obj.oracle,
        "required_inputs": list(reference_obj.required_inputs),
        "sources": [source.to_dict() for source in reference_obj.sources],
    }


def _reference_for_metric(metric: str) -> MetricReference | None:
    for candidate in _metric_reference_candidates(metric):
        reference = get_metric_reference(candidate)
        if reference is not None:
            return reference
    return None


def _section_for_reference_family(family: str) -> str:
    if family == "distribution_drift":
        return "inference_equivalence"
    return family


def _coverage_reason(report: dict[str, Any], section: str, metric: str) -> str:
    coverage = report.get("metric_coverage", {}).get("entries", [])
    if not isinstance(coverage, list):
        return ""
    for item in coverage:
        if item.get("section") == section and item.get("metric") in {
            metric,
            metric.rsplit(".", 1)[-1],
        }:
            return str(item.get("reason") or "")
    return ""


def _coverage_status(report: dict[str, Any], section: str, metric: str) -> str:
    coverage = report.get("metric_coverage", {}).get("entries", [])
    if not isinstance(coverage, list):
        return ""
    for item in coverage:
        if item.get("section") == section and item.get("metric") in {
            metric,
            metric.rsplit(".", 1)[-1],
        }:
            status = item.get("status")
            return str(status) if status else ""
    return ""


def _gate_note(report: dict[str, Any], section: str, metric: str) -> str:
    gates = report.get("gates", {})
    section_gates = gates.get("sections", {}).get(section, {}) if isinstance(gates, dict) else {}
    checks = section_gates.get("checks", {}) if isinstance(section_gates, dict) else {}
    if not isinstance(checks, dict):
        return ""
    candidates = {metric, metric.rsplit(".", 1)[-1], metric.replace(".", "_")}
    for name, check in checks.items():
        if not isinstance(check, dict):
            continue
        check_metric = str(check.get("metric_path") or check.get("metric") or name)
        if name in candidates or check_metric in candidates:
            status = check.get("status", "unknown")
            threshold = check.get("threshold", "")
            return f"gate {name}: status={status}, threshold={threshold}"
    return ""


def _section_reason(section_payload: Any) -> str:
    if not isinstance(section_payload, dict):
        return ""
    error = section_payload.get("error")
    if error:
        return str(error).replace("\n", " ")[:500]
    warnings = section_payload.get("warnings")
    if isinstance(warnings, list) and warnings:
        return "; ".join(str(item).replace("\n", " ") for item in warnings)[:500]
    return ""


def _missing_metric_status(metric: str) -> str:
    lowered = metric.lower()
    if lowered.startswith("recovery") or "error_repeat" in lowered or "fallback" in lowered:
        return "not_applicable"
    if any(token in lowered for token in ("judge", "faithfulness", "grounding")):
        return "blocked"
    if lowered.startswith("kv_") or "kv_cache" in lowered or "device_drift" in lowered:
        return "blocked"
    if any(token in lowered for token in ("delta_vs_ref", "relative_delta_vs_ref")):
        return "blocked"
    if _reference_for_metric(metric) is not None:
        return "blocked"
    return "not_collected"


def _value_metric_status(metric: str, value: Any) -> str:
    metric_name = metric.lower()
    if isinstance(value, bool):
        if metric_name.rsplit(".", 1)[-1] in {"json_valid", "schema_valid"}:
            return "pass" if value else "fail"
        return ""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return ""
    numeric = float(value)
    if _is_zero_drift_metric(metric_name):
        return "pass" if abs(numeric) <= 1e-5 else "fail"
    if _is_one_quality_metric(metric_name):
        return "pass" if numeric >= 0.999 else "fail"
    if metric_name in {"redundant_tool_call_rate", "duplicate_sentence_ratio"}:
        return "pass" if abs(numeric) <= 1e-12 else "warning"
    if metric_name in {"state_drift_score", "policy_violation_rate"}:
        return "pass" if abs(numeric) <= 1e-12 else "fail"
    if metric_name in {
        "tool_call_validity",
        "tool_correctness",
        "task_completion",
        "evidence_coverage",
        "context_recall",
    }:
        return "pass" if numeric >= 0.999 else "warning"
    return ""


def _value_metric_reason(metric: str, value: Any) -> str:
    if not _value_metric_status(metric, value):
        return ""
    metric_name = metric.lower()
    if _is_zero_drift_metric(metric_name):
        return "deterministic drift oracle: expected value is near 0 without configured gate"
    if _is_one_quality_metric(metric_name):
        return (
            "deterministic overlap/cosine oracle: expected value is near 1 without configured gate"
        )
    if metric_name in {"redundant_tool_call_rate", "duplicate_sentence_ratio"}:
        return (
            "deterministic rate oracle: lower is better; zero means no redundant/repeated behavior"
        )
    if metric_name in {"state_drift_score", "policy_violation_rate"}:
        return "deterministic rate oracle: zero means no state drift or policy violation"
    if metric_name in {
        "tool_call_validity",
        "tool_correctness",
        "task_completion",
        "evidence_coverage",
        "context_recall",
    }:
        return "deterministic fixture oracle: expected score is 1.0 for the supplied fixture"
    if metric_name.rsplit(".", 1)[-1] in {"json_valid", "schema_valid"}:
        return "deterministic parser/schema oracle"
    return ""


def _is_zero_drift_metric(metric_name: str) -> bool:
    return any(
        token in metric_name
        for token in (
            "mean_kl",
            "p95_kl",
            "max_kl",
            "mean_js",
            "p95_js",
            "batch_js",
            "kv_mean_js",
            "entropy_drift",
            "top1_changed_rate",
            "top1_change_rate",
            "prefix_divergence_rate",
        )
    )


def _is_one_quality_metric(metric_name: str) -> bool:
    return any(
        token in metric_name
        for token in (
            "top5_overlap",
            "top10_overlap",
            "topk_overlap",
            "mean_logit_cosine",
            "unique_ngram_ratio",
        )
    )


def _lookup_dotted_metric(payload: dict[str, Any], metric: str) -> tuple[str | None, Any]:
    current: Any = payload
    parts = metric.split(".")
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None, None
        current = current[part]
    return metric, current


def _metric_reference_candidates(metric: str) -> tuple[str, ...]:
    parts = tuple(part for part in metric.split(".") if part)
    if not parts:
        return (metric,)
    candidates = [metric, parts[-1]]
    if len(parts) > 1:
        candidates.insert(1, "_".join(parts))
    return tuple(dict.fromkeys(candidates))


_NON_METRIC_KEYS = {
    "id",
    "status",
    "warnings",
    "warning",
    "error",
    "reason",
    "sample_output",
    "raw",
    "state_errors",
    "errors",
    "policy_violations",
}

_METRIC_VALUE_ALIASES = {
    "topk_overlap": ("top10_overlap",),
    "ngram_repetition_rate": ("ngram_repetition.repetition_rate",),
    "repetition_rate": ("ngram_repetition.repetition_rate",),
    "unique_ngram_ratio": ("ngram_repetition.unique_ngram_ratio",),
    "json_valid": ("json_validity.json_valid",),
    "batch_invariance_mean_kl": ("batch_invariance.mean_kl",),
    "batch_mean_kl": ("batch_invariance.mean_kl",),
    "batch_p95_kl": ("batch_invariance.p95_kl",),
    "batch_js": ("batch_invariance.mean_js",),
    "batch_entropy_drift": ("batch_invariance.mean_entropy_drift",),
    "batch_top1_changed_rate": ("batch_invariance.top1_changed_rate",),
    "batch_generation_prefix_divergence": ("generation_prefix_divergence.prefix_divergence_rate",),
    "generation_prefix_divergence": ("generation_prefix_divergence.prefix_divergence_rate",),
    "prefix_divergence_rate": ("generation_prefix_divergence.prefix_divergence_rate",),
    "kv_cache_mean_kl": ("kv_cache_drift.mean_kl",),
    "kv_cache_p95_kl": ("kv_cache_drift.p95_kl",),
    "kv_mean_kl": ("kv_cache_drift.mean_kl",),
    "kv_p95_kl": ("kv_cache_drift.p95_kl",),
    "kv_mean_js": ("kv_cache_drift.mean_js",),
    "kv_entropy_drift": ("kv_cache_drift.mean_entropy_drift",),
    "kv_top1_change_rate": ("kv_cache_drift.top1_changed_rate",),
    "context_recall": ("evidence_coverage",),
    "tool_correctness": ("tool_call_validity",),
    "agent_state_drift": ("state_drift_score",),
    "recovery_score": ("recovery_after_tool_error",),
}


__all__ = [
    "DEFAULT_REFERENCE_COMPARISON_METRICS",
    "ReferenceComparisonService",
    "collect_metric_coverage_specs",
    "discover_report_metric_specs",
    "find_metric_reference",
    "find_metric_value",
    "format_metric_value",
    "metric_status",
    "metric_reason",
    "supported_metric_specs",
]
