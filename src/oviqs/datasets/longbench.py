from __future__ import annotations

from typing import Any

from oviqs.core.sample import EvalSample


def longbench_row_to_sample(row: dict[str, Any], dataset_name: str | None = None) -> EvalSample:
    prompt = row.get("input") or row.get("question") or row.get("prompt") or ""
    context = row.get("context") or row.get("passage") or row.get("document") or ""
    answers = row.get("answers") or row.get("answer") or []
    references = [answers] if isinstance(answers, str) else [str(item) for item in answers]
    return EvalSample(
        id=str(row.get("_id") or row.get("id") or row.get("sample_id")),
        task_type="long_context",
        prompt=str(prompt),
        context=str(context),
        target=references[0] if references else str(row.get("target") or ""),
        references=references,
        metadata={"source": "longbench", "dataset": dataset_name, **row.get("metadata", {})},
    )


def longbench_rows_to_samples(
    rows: list[dict[str, Any]],
    dataset_name: str | None = None,
) -> list[EvalSample]:
    return [longbench_row_to_sample(row, dataset_name=dataset_name) for row in rows]
