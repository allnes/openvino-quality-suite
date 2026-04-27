from __future__ import annotations

from typing import Any

from oviqs.core.errors import OptionalDependencyError
from oviqs.core.sample import EvalSample
from oviqs.integrations.base import IntegrationResult, normalize_external_output, unavailable_result


def ensure_ragas_available():
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise OptionalDependencyError("ragas", "rag") from exc


def build_ragas_rows(samples: list[EvalSample], answer_rows: list[dict[str, Any]]) -> list[dict]:
    answer_by_id = {str(row.get("id")): row for row in answer_rows}
    rows = []
    for sample in samples:
        answer = answer_by_id.get(sample.id, {})
        rows.append(
            {
                "id": sample.id,
                "user_input": sample.prompt or sample.text or sample.context or "",
                "response": answer.get("answer") or sample.metadata.get("answer") or "",
                "retrieved_contexts": sample.retrieved_contexts,
                "reference": answer.get("reference")
                or sample.metadata.get("reference")
                or (sample.references[0] if sample.references else ""),
                "reference_contexts": sample.expected_evidence,
            }
        )
    return rows


def evaluate_with_ragas(
    rows: list[dict[str, Any]],
    metrics: list[Any] | None = None,
) -> IntegrationResult:
    try:
        from datasets import Dataset
        from ragas import evaluate
    except ImportError:
        return unavailable_result("ragas", "ragas", "rag")

    try:
        result = evaluate(Dataset.from_list(rows), metrics=metrics)
    except Exception as exc:
        return IntegrationResult(name="ragas", status="fail", error=str(exc))
    return IntegrationResult(
        name="ragas",
        status="pass",
        metrics=normalize_external_output(result),
        raw=result,
    )


def evaluate_rag_samples_with_ragas(
    samples: list[EvalSample],
    answer_rows: list[dict[str, Any]],
    metrics: list[Any] | None = None,
) -> IntegrationResult:
    return evaluate_with_ragas(build_ragas_rows(samples, answer_rows), metrics=metrics)
