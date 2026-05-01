from __future__ import annotations

from typing import Any

from oviqs.references import get_metric_reference

StatusOrder = {"pass": 0, "unknown": 1, "warning": 2, "fail": 3}  # nosec B105


def evaluate_gates(
    metrics: dict[str, Any],
    gates: dict[str, Any],
    *,
    require_references: bool = True,
) -> dict[str, Any]:
    """Evaluate threshold gates.

    Supported suffixes: `_max`, `_min`, `abs_..._max`. Missing metrics produce `unknown`.
    Exceeded thresholds produce `warning`; callers may promote configured critical gates to fail.
    Metrics without a registered reference/oracle also produce `unknown` by default.
    """

    sections: dict[str, Any] = {}
    overall = "pass"
    metric_references = metrics.get("metric_references", {})
    for section, thresholds in gates.items():
        section_metrics = metrics.get(section, {})
        checks = {}
        section_status = "pass"
        for gate_name, threshold in thresholds.items():
            metric_name, mode = _parse_gate_name(gate_name)
            value, metric_path = _find_metric_value(section_metrics, metric_name)
            reference = _find_metric_reference(
                metric_references,
                section,
                metric_name,
                metric_path,
            )
            if value is None or (require_references and reference is None):
                status = "unknown"
                passed = None
            else:
                passed = _passes(float(value), float(threshold), mode)
                status = "pass" if passed else "warning"
            checks[gate_name] = {
                "metric": metric_name,
                "metric_path": metric_path,
                "value": value,
                "threshold": threshold,
                "reference_status": "present" if reference is not None else "missing",
                "status": status,
            }
            if reference is not None:
                checks[gate_name]["reference"] = reference
            elif value is not None and require_references:
                checks[gate_name]["warning"] = "metric has no registered reference/oracle"
            section_status = _worst(section_status, status)
        sections[section] = {"status": section_status, "checks": checks}
        overall = _worst(overall, section_status)
    return {"overall_status": overall, "sections": sections}


def _parse_gate_name(name: str) -> tuple[str, str]:
    if name.startswith("abs_") and name.endswith("_max"):
        return name.removeprefix("abs_").removesuffix("_max"), "abs_max"
    if name.endswith("_max"):
        return name.removesuffix("_max"), "max"
    if name.endswith("_min"):
        return name.removesuffix("_min"), "min"
    raise ValueError(f"Unsupported gate naming convention: {name}")


def _passes(value: float, threshold: float, mode: str) -> bool:
    if mode == "max":
        return value <= threshold
    if mode == "min":
        return value >= threshold
    if mode == "abs_max":
        return abs(value) <= threshold
    raise ValueError(f"Unsupported gate mode: {mode}")


def _find_metric_value(section_metrics: Any, metric_name: str) -> tuple[Any, str | None]:
    if not isinstance(section_metrics, dict):
        return None, None
    if metric_name in section_metrics:
        return section_metrics[metric_name], metric_name
    for path, leaf_name, value in _iter_scalar_metrics(section_metrics):
        if leaf_name == metric_name or path.replace(".", "_") == metric_name:
            return value, path
    return None, None


def _iter_scalar_metrics(payload: dict[str, Any], prefix: str = ""):
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _iter_scalar_metrics(value, path)
        elif not isinstance(value, list):
            yield path, key, value


def _find_metric_reference(
    metric_references: Any,
    section: str,
    metric_name: str,
    metric_path: str | None,
) -> dict[str, Any] | None:
    if isinstance(metric_references, dict):
        section_refs = metric_references.get(section, {})
        if isinstance(section_refs, dict):
            for key in _reference_lookup_keys(metric_name, metric_path):
                reference = section_refs.get(key)
                if isinstance(reference, dict):
                    return reference
    reference = get_metric_reference(metric_name)
    if reference is not None:
        return reference.to_dict()
    if metric_path:
        path_parts = metric_path.split(".")
        for key in (path_parts[0], path_parts[-1]):
            reference = get_metric_reference(key)
            if reference is not None:
                return reference.to_dict()
    return None


def _reference_lookup_keys(metric_name: str, metric_path: str | None) -> list[str]:
    keys = [metric_name]
    if metric_path:
        keys.append(metric_path)
        keys.extend(metric_path.split("."))
    return list(dict.fromkeys(keys))


def _worst(left: str, right: str) -> str:
    return left if StatusOrder[left] >= StatusOrder[right] else right
