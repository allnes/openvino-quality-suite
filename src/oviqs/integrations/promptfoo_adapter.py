from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oviqs.integrations.base import IntegrationResult, normalize_external_output


def import_promptfoo_results(path: str | Path) -> IntegrationResult:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        return IntegrationResult(name="promptfoo", status="fail", error=str(exc))
    return IntegrationResult(
        name="promptfoo",
        status="pass",
        metrics=normalize_external_output(payload),
        raw=payload,
    )


def evaluate_with_promptfoo_python(evaluator, test_cases: list[dict[str, Any]], **kwargs):
    from oviqs.integrations.base import run_callable_integration

    return run_callable_integration("promptfoo", evaluator, test_cases=test_cases, **kwargs)
