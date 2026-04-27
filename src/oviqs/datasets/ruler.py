from __future__ import annotations

from typing import Any

from oviqs.core.sample import EvalSample


def ruler_row_to_sample(row: dict[str, Any], task_name: str | None = None) -> EvalSample:
    prompt = row.get("input") or row.get("prompt") or row.get("query") or ""
    context = row.get("context") or row.get("haystack") or ""
    target = row.get("outputs") or row.get("answer") or row.get("target") or ""
    if isinstance(target, list):
        references = [str(item) for item in target]
        target_text = references[0] if references else ""
    else:
        target_text = str(target)
        references = [target_text] if target_text else []
    return EvalSample(
        id=str(row.get("id") or row.get("index") or row.get("sample_id")),
        task_type="long_context",
        prompt=str(prompt),
        context=str(context),
        target=target_text,
        references=references,
        metadata={"source": "ruler", "task": task_name, **row.get("metadata", {})},
    )


def ruler_rows_to_samples(
    rows: list[dict[str, Any]],
    task_name: str | None = None,
) -> list[EvalSample]:
    return [ruler_row_to_sample(row, task_name=task_name) for row in rows]
