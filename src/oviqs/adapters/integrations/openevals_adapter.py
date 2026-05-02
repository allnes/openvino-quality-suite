from __future__ import annotations

from typing import Any

from oviqs.adapters.integrations.base import (
    IntegrationResult,
    run_callable_integration,
    unavailable_result,
)


def ensure_openevals_available():
    try:
        import openevals  # noqa: F401
    except ImportError:
        return unavailable_result("openevals", "openevals", "agent")
    return IntegrationResult(name="openevals", status="pass")


def evaluate_with_openevals(
    evaluator,
    inputs: Any,
    outputs: Any,
    reference_outputs: Any | None = None,
    **kwargs,
) -> IntegrationResult:
    payload = {"inputs": inputs, "outputs": outputs, **kwargs}
    if reference_outputs is not None:
        payload["reference_outputs"] = reference_outputs
    return run_callable_integration("openevals", evaluator, **payload)


def evaluate_with_agentevals(
    evaluator,
    trajectory: list[dict[str, Any]],
    **kwargs,
) -> IntegrationResult:
    return run_callable_integration("agentevals", evaluator, trajectory=trajectory, **kwargs)
