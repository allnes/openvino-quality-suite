from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from oviqs.domain.references import get_metric_reference
from oviqs.domain.reporting import MetricObservation, MetricPath, status_to_severity
from oviqs.domain.reporting.severity import ReportStatus

SCALAR_TYPES = (int, float, str, bool, type(None))
REPORT_SECTIONS = (
    "inference_equivalence",
    "likelihood",
    "long_context",
    "generation",
    "rag",
    "agent",
    "serving",
    "performance",
    "reproducibility",
)


def flatten_report_metrics(
    report: dict[str, Any],
    *,
    baseline: dict[str, Any] | None = None,
    gates: dict[str, Any] | None = None,
) -> list[MetricObservation]:
    observations: dict[str, MetricObservation] = {}
    gate_index = _gate_index(report.get("gates") or gates or {})

    for section in REPORT_SECTIONS:
        payload = report.get(section)
        if isinstance(payload, dict):
            section_status = _normalize_status(payload.get("status", "pass"))
            for parts, value in _walk_scalars(payload):
                if parts == ("status",):
                    continue
                path = MetricPath.from_parts(section, parts)
                baseline_value = _lookup_baseline_scalar(baseline, section, parts)
                gate = gate_index.get(path.dotted)
                status = _observation_status(section_status, gate)
                observations[path.dotted] = _build_observation(
                    path,
                    value,
                    status=status,
                    gate=gate,
                    baseline_value=baseline_value,
                )

    for path_text, gate in gate_index.items():
        if path_text in observations:
            continue
        try:
            path = MetricPath.parse(path_text)
        except ValueError:
            continue
        observations[path_text] = _build_observation(
            path,
            None,
            status="unknown",
            gate=gate,
            baseline_value=_lookup_baseline_scalar(baseline, path.section, path.parts),
        )

    coverage_entries = report.get("metric_coverage", {}).get("entries", [])
    if isinstance(coverage_entries, list):
        for entry in coverage_entries:
            if not isinstance(entry, dict):
                continue
            section = str(entry.get("section") or "")
            metric = str(entry.get("metric") or "")
            if not section or not metric:
                continue
            path_text = f"{section}.{metric}"
            if path_text in observations:
                continue
            status = "unknown" if entry.get("status") == "unknown" else "pass"
            observations[path_text] = _build_observation(
                MetricPath(section=section, name=metric, parts=(metric,)),
                entry.get("value"),
                status=status,
                gate=None,
                baseline_value=None,
                reason=str(entry.get("reason") or ""),
            )

    return sorted(observations.values(), key=lambda item: item.path)


def _walk_scalars(
    payload: dict[str, Any],
    prefix: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], Any]]:
    for key, value in payload.items():
        path = (*prefix, str(key))
        if isinstance(value, SCALAR_TYPES):
            yield path, value
        elif isinstance(value, dict):
            yield from _walk_scalars(value, path)


def _build_observation(
    path: MetricPath,
    value: Any,
    *,
    status: ReportStatus,
    gate: dict[str, Any] | None,
    baseline_value: float | int | None,
    reason: str = "",
) -> MetricObservation:
    reference = get_metric_reference(path.name)
    numeric_value = (
        value if isinstance(value, int | float) and not isinstance(value, bool) else None
    )
    delta_abs = None
    delta_rel = None
    if isinstance(numeric_value, int | float) and isinstance(baseline_value, int | float):
        delta_abs = float(numeric_value) - float(baseline_value)
        if baseline_value != 0:
            delta_rel = delta_abs / abs(float(baseline_value))
    threshold = None
    if gate:
        raw_threshold = gate.get("threshold")
        threshold = raw_threshold if isinstance(raw_threshold, int | float) else None
    tags = tuple(item for item in [reason or None, "gated" if gate else None] if item)
    return MetricObservation(
        path=path.dotted,
        section=path.section,
        name=path.name,
        value=value if isinstance(value, SCALAR_TYPES) else None,
        unit=_unit_for(path.name),
        status=status,
        severity=status_to_severity(status),
        reference_id=reference.primary_reference if reference else None,
        degradation_rule=reference.degradation_rule if reference else None,
        baseline_value=baseline_value,
        delta_abs=delta_abs,
        delta_rel=delta_rel,
        threshold=threshold,
        threshold_rule=_threshold_rule(gate),
        sample_count=_sample_count_hint(path),
        tags=tags,
    )


def _gate_index(gates: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    sections = gates.get("sections") if isinstance(gates, dict) else None
    if not isinstance(sections, dict):
        return indexed
    for section, payload in sections.items():
        checks = payload.get("checks") if isinstance(payload, dict) else None
        if not isinstance(checks, dict):
            continue
        for metric, check in checks.items():
            if isinstance(check, dict):
                metric_path = check.get("metric_path") or check.get("metric") or metric
                indexed[f"{section}.{metric_path}"] = check
    return indexed


def _observation_status(section_status: ReportStatus, gate: dict[str, Any] | None) -> ReportStatus:
    if gate:
        return _normalize_status(gate.get("status", "unknown"))
    return section_status


def _normalize_status(value: Any) -> ReportStatus:
    if value in {"pass", "warning", "fail", "unknown"}:
        return value
    return "unknown"


def _lookup_baseline_scalar(
    baseline: dict[str, Any] | None,
    section: str,
    parts: tuple[str, ...],
) -> float | int | None:
    if not baseline:
        return None
    value: Any = baseline.get(section)
    for part in parts:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int | float) else None


def _unit_for(metric_name: str) -> str | None:
    if (
        metric_name.endswith("_rate")
        or metric_name.endswith("_ratio")
        or metric_name.endswith("_validity")
    ):
        return "ratio"
    if metric_name.endswith("_ms") or metric_name.endswith("_latency_ms"):
        return "ms"
    if metric_name.endswith("_tokens") or metric_name == "num_tokens":
        return "tokens"
    return None


def _threshold_rule(gate: dict[str, Any] | None) -> str | None:
    if not gate:
        return None
    if gate.get("rule"):
        return str(gate["rule"])
    threshold = gate.get("threshold")
    if threshold is None:
        return None
    return f"threshold={threshold}"


def _sample_count_hint(path: MetricPath) -> int | None:
    if path.name in {"samples", "per_token", "entries"}:
        return None
    return None


__all__ = ["REPORT_SECTIONS", "flatten_report_metrics"]
