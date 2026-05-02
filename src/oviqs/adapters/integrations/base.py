from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from oviqs.adapters.errors import OptionalDependencyError


@dataclass
class IntegrationResult:
    name: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    raw: Any | None = None

    def to_report_section(self) -> dict[str, Any]:
        section = {"status": self.status, **self.metrics}
        if self.samples:
            section["samples"] = self.samples
        if self.warnings:
            section["warnings"] = self.warnings
        if self.error:
            section["error"] = self.error
        return section


def optional_module(module_name: str, extra: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise OptionalDependencyError(module_name, extra) from exc


def unavailable_result(name: str, package: str, extra: str) -> IntegrationResult:
    return IntegrationResult(
        name=name,
        status="unavailable",
        warnings=[
            f"Missing optional dependency '{package}'. Install with: pip install 'oviqs[{extra}]'"
        ],
    )


def normalize_external_output(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_pandas"):
        return {"data": value.to_pandas().to_dict(orient="records")}
    if hasattr(value, "to_dict"):
        try:
            converted = value.to_dict()
            if isinstance(converted, dict):
                return converted
            return {"data": converted}
        except TypeError:
            pass
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {"raw": repr(value)}


def run_callable_integration(
    name: str,
    evaluator: Callable[..., Any],
    *args,
    **kwargs,
) -> IntegrationResult:
    try:
        output = evaluator(*args, **kwargs)
    except Exception as exc:
        return IntegrationResult(name=name, status="fail", error=str(exc))
    return IntegrationResult(
        name=name,
        status="pass",
        metrics=normalize_external_output(output),
        raw=output,
    )
