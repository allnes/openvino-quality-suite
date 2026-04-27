from __future__ import annotations

from typing import Any

from oviqs.integrations.base import IntegrationResult, normalize_external_output, unavailable_result


def evaluate_with_lm_eval(
    model: Any,
    tasks: list[str],
    model_args: str | dict[str, Any] | None = None,
    num_fewshot: int | None = None,
    batch_size: int | str | None = None,
    **kwargs,
) -> IntegrationResult:
    try:
        from lm_eval import simple_evaluate
    except ImportError:
        return unavailable_result("lm-evaluation-harness", "lm-eval", "eval")

    payload = {"model": model, "tasks": tasks, **kwargs}
    if model_args is not None:
        payload["model_args"] = model_args
    if num_fewshot is not None:
        payload["num_fewshot"] = num_fewshot
    if batch_size is not None:
        payload["batch_size"] = batch_size
    try:
        result = simple_evaluate(**payload)
    except Exception as exc:
        return IntegrationResult(name="lm-eval", status="fail", error=str(exc))
    return IntegrationResult(
        name="lm-eval",
        status="pass",
        metrics=normalize_external_output(result),
        raw=result,
    )
