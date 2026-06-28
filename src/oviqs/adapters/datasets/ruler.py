from __future__ import annotations

from pathlib import Path
from typing import Any

from oviqs.adapters.datasets.jsonl import read_jsonl, write_jsonl
from oviqs.adapters.errors import OptionalDependencyError
from oviqs.domain.samples import EvalSample


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


# Recommended open long-context dataset for the long_context family
# (RULER-style synthetic length/position stress). simonjegou/ruler exposes one
# config per context length ("4096", "8192", ...) with a "test" split.
RULER_HF_ID = "simonjegou/ruler"
RULER_DEFAULT_CONFIG = "4096"


def load_ruler_samples(
    config: str = RULER_DEFAULT_CONFIG,
    split: str = "test",
    limit: int = 16,
    cache_path: str | Path | None = None,
    task_name: str | None = None,
) -> list[EvalSample]:
    """Load RULER long-context rows from Hugging Face into EvalSample objects.

    Downloads ``simonjegou/ruler`` (config = context length) via the optional
    ``datasets`` dependency, converts rows with :func:`ruler_row_to_sample`, and
    caches the raw rows to ``cache_path`` (JSONL) for offline reuse.
    """

    if cache_path is not None and Path(cache_path).exists():
        rows = read_jsonl(cache_path)[:limit]
        return ruler_rows_to_samples(rows, task_name=task_name or config)
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without datasets
        raise OptionalDependencyError("datasets", "openvino") from exc
    dataset = load_dataset(RULER_HF_ID, config, split=f"{split}[:{limit}]")
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        row = dict(item)
        row.setdefault("id", f"ruler_{config}_{idx:03d}")
        rows.append(row)
    if cache_path is not None:
        write_jsonl(rows, cache_path)
    return ruler_rows_to_samples(rows, task_name=task_name or config)
