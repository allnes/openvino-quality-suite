from __future__ import annotations

from pathlib import Path
from typing import Any

from oviqs.adapters.datasets.jsonl import load_jsonl_samples, read_jsonl, write_jsonl
from oviqs.adapters.errors import OptionalDependencyError
from oviqs.domain.samples import EvalSample


def ragas_row_to_sample(row: dict[str, Any]) -> EvalSample:
    """Convert a Ragas-style RAG row (question/contexts/ground_truth) to a sample.

    Maps the question to ``prompt``, retrieved passages to ``retrieved_contexts``
    and the reference answer to both ``target`` and ``expected_evidence`` so the
    rule-based retrieval / grounding / citation metrics have ground truth.
    """

    question = row.get("question") or row.get("query") or row.get("user_input") or ""
    contexts = row.get("contexts") or row.get("retrieved_contexts") or row.get("context") or []
    if isinstance(contexts, str):
        contexts = [contexts]
    ground_truth = (
        row.get("ground_truth")
        or row.get("ground_truths")
        or row.get("reference")
        or row.get("answer")
        or ""
    )
    if isinstance(ground_truth, list):
        references = [str(item) for item in ground_truth]
    else:
        references = [str(ground_truth)] if ground_truth else []
    answer = row.get("answer") or row.get("response") or ""
    return EvalSample(
        id=str(row.get("id") or row.get("_id") or row.get("sample_id")),
        task_type="rag",
        prompt=str(question),
        target=references[0] if references else "",
        references=references,
        retrieved_contexts=[str(item) for item in contexts],
        expected_evidence=references,
        metadata={"source": "ragas_amnesty_qa", "answer": str(answer)},
    )


def ragas_rows_to_samples(rows: list[dict[str, Any]]) -> list[EvalSample]:
    return [ragas_row_to_sample(row) for row in rows]


# Recommended open RAG-evaluation dataset for the rag family (Ragas-style).
AMNESTY_QA_HF_ID = "explodinggradients/amnesty_qa"
AMNESTY_QA_DEFAULT_CONFIG = "english_v3"


def load_ragas_amnesty_samples(
    config: str = AMNESTY_QA_DEFAULT_CONFIG,
    split: str = "eval",
    limit: int = 16,
    cache_path: str | Path | None = None,
) -> list[EvalSample]:
    """Load the Ragas amnesty_qa RAG dataset from Hugging Face into EvalSamples.

    Uses the optional ``datasets`` dependency and caches raw rows to
    ``cache_path`` (JSONL) for offline reuse.
    """

    if cache_path is not None and Path(cache_path).exists():
        rows = read_jsonl(cache_path)[:limit]
        return ragas_rows_to_samples(rows)
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without datasets
        raise OptionalDependencyError("datasets", "rag") from exc
    dataset = load_dataset(AMNESTY_QA_HF_ID, config, split=f"{split}[:{limit}]")
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        row = dict(item)
        row.setdefault("id", f"amnesty_qa_{idx:03d}")
        rows.append(row)
    if cache_path is not None:
        write_jsonl(rows, cache_path)
    return ragas_rows_to_samples(rows)


__all__ = [
    "AMNESTY_QA_HF_ID",
    "load_jsonl_samples",
    "load_ragas_amnesty_samples",
    "ragas_row_to_sample",
    "ragas_rows_to_samples",
]
