from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oviqs.application.reporting.schema_normalization import (
    UnsupportedReportSchemaVersion,
    normalize_evaluation_report_contract,
)
from oviqs.domain.reporting.severity import ReportStatus, Severity

VALID_STATUSES: set[str] = {"pass", "warning", "fail", "unknown"}
VALID_SEVERITIES: set[str] = {"none", "low", "medium", "high", "critical"}
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "contracts" / "jsonschema"


def validate_evaluation_report_contract(report: dict[str, Any]) -> list[str]:
    try:
        report = normalize_evaluation_report_contract(report)
    except UnsupportedReportSchemaVersion as exc:
        return [str(exc)]
    errors = validate_with_project_schema(report, "evaluation_report.schema.json")
    _require_mapping(report, "report", errors)
    _require_string(report, "schema_version", errors)
    run = _require_mapping(report.get("run"), "run", errors)
    if run is not None:
        _require_string(run, "id", errors, parent="run")
        _require_string(run, "suite", errors, parent="run")
        _require_string(run, "created_at", errors, parent="run")
    summary = _require_mapping(report.get("summary"), "summary", errors)
    if summary is not None:
        status = summary.get("overall_status")
        if status not in VALID_STATUSES:
            errors.append("summary.overall_status must be one of pass, warning, fail, unknown")
    analysis = report.get("analysis")
    if isinstance(analysis, dict):
        metrics = analysis.get("metrics", [])
        if not isinstance(metrics, list):
            errors.append("analysis.metrics must be an array when present")
        else:
            for idx, metric in enumerate(metrics):
                errors.extend(
                    validate_metric_observation_contract(metric, f"analysis.metrics[{idx}]")
                )
        findings = analysis.get("findings", [])
        if not isinstance(findings, list):
            errors.append("analysis.findings must be an array when present")
    return errors


def validate_with_project_schema(payload: Any, schema_name: str) -> list[str]:
    schema_path = SCHEMA_DIR / schema_name
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema_errors = _validate_with_jsonschema(payload, schema)
    if jsonschema_errors is not None:
        return jsonschema_errors
    return list(_validate_schema(payload, schema, "$"))


def _validate_with_jsonschema(payload: Any, schema: dict[str, Any]) -> list[str] | None:
    try:
        from jsonschema import Draft202012Validator  # type: ignore[reportMissingModuleSource]
    except ImportError:
        return None
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.absolute_path))
    return [f"{_format_json_path(error.absolute_path)}: {error.message}" for error in errors]


def _format_json_path(path: Any) -> str:
    parts = list(path)
    if not parts:
        return "$"
    rendered = "$"
    for part in parts:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += f".{part}"
    return rendered


def _validate_schema(payload: Any, schema: dict[str, Any], path: str):
    schema_type = schema.get("type")
    if schema_type is not None and not _matches_schema_type(payload, schema_type):
        yield f"{path} must be {_type_description(schema_type)}"
        return
    enum = schema.get("enum")
    if enum is not None and payload not in enum:
        yield f"{path} must be one of {', '.join(str(item) for item in enum)}"
    if isinstance(payload, dict):
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in payload:
                    yield f"{path}.{key} is required"
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, child_schema in properties.items():
                if key in payload and isinstance(child_schema, dict):
                    yield from _validate_schema(payload[key], child_schema, f"{path}.{key}")
    if isinstance(payload, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(payload):
                yield from _validate_schema(item, item_schema, f"{path}[{idx}]")


def _matches_schema_type(payload: Any, schema_type: Any) -> bool:
    if isinstance(schema_type, list):
        return any(_matches_schema_type(payload, item) for item in schema_type)
    if schema_type == "object":
        return isinstance(payload, dict)
    if schema_type == "array":
        return isinstance(payload, list)
    if schema_type == "string":
        return isinstance(payload, str)
    if schema_type == "number":
        return isinstance(payload, int | float) and not isinstance(payload, bool)
    if schema_type == "integer":
        return isinstance(payload, int) and not isinstance(payload, bool)
    if schema_type == "boolean":
        return isinstance(payload, bool)
    if schema_type == "null":
        return payload is None
    return True


def _type_description(schema_type: Any) -> str:
    if isinstance(schema_type, list):
        return " or ".join(str(item) for item in schema_type)
    return str(schema_type)


def validate_metric_observation_contract(metric: Any, prefix: str = "metric") -> list[str]:
    errors: list[str] = []
    if not isinstance(metric, dict):
        return [f"{prefix} must be an object"]
    for key in ("path", "section", "name"):
        if not isinstance(metric.get(key), str) or not metric.get(key):
            errors.append(f"{prefix}.{key} must be a non-empty string")
    if metric.get("status") not in VALID_STATUSES:
        errors.append(f"{prefix}.status must be one of pass, warning, fail, unknown")
    if metric.get("severity") not in VALID_SEVERITIES:
        errors.append(f"{prefix}.severity must be one of none, low, medium, high, critical")
    tags = metric.get("tags", [])
    if tags is not None and not (
        isinstance(tags, list) and all(isinstance(item, str) for item in tags)
    ):
        errors.append(f"{prefix}.tags must be an array of strings")
    return errors


def validate_sample_metric_contract(row: Any, prefix: str = "sample_metric") -> list[str]:
    errors = validate_with_project_schema(row, "sample_metric.schema.json")
    if not isinstance(row, dict):
        return errors
    if not isinstance(row.get("section"), str) or not row.get("section"):
        errors.append(f"{prefix}.section must be a non-empty string")
    if not isinstance(row.get("sample_index"), int) or isinstance(row.get("sample_index"), bool):
        errors.append(f"{prefix}.sample_index must be an integer")
    status = row.get("status")
    if status is not None and status not in VALID_STATUSES:
        errors.append(f"{prefix}.status must be one of pass, warning, fail, unknown")
    return errors


def validate_sample_metrics_contract(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for idx, row in enumerate(rows):
        errors.extend(validate_sample_metric_contract(row, f"sample_metrics[{idx}]"))
    return errors


def validate_report_bundle_metadata(bundle: dict[str, Any]) -> list[str]:
    errors = validate_with_project_schema(bundle, "report_bundle.schema.json")
    for key in (
        "root",
        "report_json",
        "analysis_json",
        "metrics_csv",
        "sample_metrics_jsonl",
        "index_md",
        "dashboard_html",
    ):
        if not isinstance(bundle.get(key), str) or not bundle.get(key):
            errors.append(f"{key} must be a non-empty string")
    metadata_json = bundle.get("metadata_json")
    if metadata_json is not None and not (isinstance(metadata_json, str) and metadata_json):
        errors.append("metadata_json must be a non-empty string when present")
    return errors


def _require_mapping(value: Any, path: str, errors: list[str]) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        errors.append(f"{path} must be an object")
        return None
    return value


def _require_string(
    payload: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    parent: str | None = None,
) -> None:
    path = f"{parent}.{key}" if parent else key
    if not isinstance(payload.get(key), str) or not payload.get(key):
        errors.append(f"{path} must be a non-empty string")


__all__ = [
    "ReportStatus",
    "Severity",
    "validate_evaluation_report_contract",
    "validate_metric_observation_contract",
    "validate_report_bundle_metadata",
    "validate_sample_metric_contract",
    "validate_sample_metrics_contract",
    "validate_with_project_schema",
]
