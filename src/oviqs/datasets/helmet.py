from __future__ import annotations

from typing import Any

from oviqs.core.sample import EvalSample


def helmet_row_to_sample(row: dict[str, Any], task_name: str | None = None) -> EvalSample:
    prompt = row.get("prompt") or row.get("input") or row.get("question") or ""
    context = row.get("context") or row.get("documents") or row.get("passage") or ""
    if isinstance(context, list):
        context = "\n".join(str(item) for item in context)
    target = row.get("target") or row.get("answer") or row.get("reference") or ""
    return EvalSample(
        id=str(row.get("id") or row.get("_id") or row.get("sample_id")),
        task_type="long_context",
        prompt=str(prompt),
        context=str(context),
        target=str(target),
        references=[str(target)] if target else [],
        metadata={"source": "helmet", "task": task_name, **row.get("metadata", {})},
    )


def helmet_rows_to_samples(
    rows: list[dict[str, Any]],
    task_name: str | None = None,
) -> list[EvalSample]:
    return [helmet_row_to_sample(row, task_name=task_name) for row in rows]
