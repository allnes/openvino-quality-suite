from __future__ import annotations

from typing import Any

from oviqs.adapters.errors import OptionalDependencyError
from oviqs.adapters.integrations.base import (
    IntegrationResult,
    normalize_external_output,
    unavailable_result,
)


def ensure_deepeval_available():
    try:
        import deepeval  # noqa: F401
    except ImportError as exc:
        raise OptionalDependencyError("deepeval", "agent") from exc


def build_llm_test_case(
    input: str,
    actual_output: str,
    expected_output: str | None = None,
    retrieval_context: list[str] | None = None,
    tools_called: list[Any] | None = None,
):
    try:
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise OptionalDependencyError("deepeval", "agent") from exc

    kwargs = {
        "input": input,
        "actual_output": actual_output,
        "expected_output": expected_output,
        "retrieval_context": retrieval_context,
    }
    if tools_called is not None:
        kwargs["tools_called"] = tools_called
    return LLMTestCase(**{key: value for key, value in kwargs.items() if value is not None})


def evaluate_with_deepeval(test_cases: list[Any], metrics: list[Any]) -> IntegrationResult:
    try:
        from deepeval import evaluate
    except ImportError:
        return unavailable_result("deepeval", "deepeval", "agent")

    try:
        result = evaluate(test_cases=test_cases, metrics=metrics)
    except TypeError:
        try:
            result = evaluate(test_cases, metrics)
        except Exception as exc:
            return IntegrationResult(name="deepeval", status="fail", error=str(exc))
    except Exception as exc:
        return IntegrationResult(name="deepeval", status="fail", error=str(exc))
    return IntegrationResult(
        name="deepeval",
        status="pass",
        metrics=normalize_external_output(result),
        raw=result,
    )
