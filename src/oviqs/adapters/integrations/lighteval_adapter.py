from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oviqs.adapters.integrations.base import (
    IntegrationResult,
    normalize_external_output,
    unavailable_result,
)


@dataclass
class LightEvalTask:
    name: str
    suite: str | None = None
    fewshot: int | None = None

    @property
    def task_string(self) -> str:
        parts = [self.suite, self.name, str(self.fewshot) if self.fewshot is not None else None]
        return "|".join(part for part in parts if part)


def ensure_lighteval_available() -> IntegrationResult:
    try:
        import lighteval  # noqa: F401
    except ImportError:
        return unavailable_result("lighteval", "lighteval", "eval")
    return IntegrationResult(name="lighteval", status="pass")


def evaluate_with_lighteval(
    pipeline: Any,
    tasks: list[LightEvalTask] | list[str],
    **kwargs,
) -> IntegrationResult:
    task_values = [task.task_string if isinstance(task, LightEvalTask) else task for task in tasks]
    try:
        if hasattr(pipeline, "evaluate"):
            result = pipeline.evaluate(tasks=task_values, **kwargs)
        elif hasattr(pipeline, "run"):
            result = pipeline.run(tasks=task_values, **kwargs)
        else:
            return IntegrationResult(
                name="lighteval",
                status="fail",
                error="LightEval pipeline must expose evaluate(...) or run(...).",
            )
    except Exception as exc:
        return IntegrationResult(name="lighteval", status="fail", error=str(exc))
    return IntegrationResult(
        name="lighteval",
        status="pass",
        metrics=normalize_external_output(result),
        raw=result,
    )
