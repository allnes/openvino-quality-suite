from __future__ import annotations

from pathlib import Path

from oviqs.integrations.base import (
    IntegrationResult,
    normalize_external_output,
    run_callable_integration,
    unavailable_result,
)
from oviqs.reporting.json_report import load_report


def ensure_opencompass_available() -> IntegrationResult:
    try:
        import opencompass  # noqa: F401
    except ImportError:
        return unavailable_result("opencompass", "opencompass", "eval")
    return IntegrationResult(name="opencompass", status="pass")


def evaluate_with_opencompass(runner, config: str | Path, **kwargs) -> IntegrationResult:
    return run_callable_integration("opencompass", runner, config=str(config), **kwargs)


def import_opencompass_report(path: str | Path) -> IntegrationResult:
    try:
        payload = load_report(Path(path))
    except Exception as exc:
        return IntegrationResult(name="opencompass", status="fail", error=str(exc))
    return IntegrationResult(
        name="opencompass",
        status="pass",
        metrics=normalize_external_output(payload),
        raw=payload,
    )
