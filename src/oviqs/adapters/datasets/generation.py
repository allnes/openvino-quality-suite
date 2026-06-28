from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oviqs.adapters.datasets.jsonl import read_jsonl, write_jsonl
from oviqs.adapters.errors import OptionalDependencyError
from oviqs.domain.samples import EvalSample


def jsonschema_row_to_sample(row: dict[str, Any], subset: str | None = None) -> EvalSample:
    """Convert a JSONSchemaBench row into a structured-output generation sample.

    The JSON schema the output must satisfy is stored in
    ``expected_constraints["json_schema"]`` so the generation metrics can run
    schema validity / required-section checks against it.
    """

    schema = row.get("json_schema") or row.get("schema")
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            schema = {"raw": schema}
    prompt = row.get("prompt") or row.get("input") or ""
    if not prompt and isinstance(schema, dict):
        prompt = f"Return a JSON object that conforms to this schema: {json.dumps(schema)}"
    return EvalSample(
        id=str(row.get("id") or row.get("unique_id") or row.get("_id")),
        task_type="generation",
        prompt=str(prompt),
        expected_constraints={"json_schema": schema} if schema is not None else {},
        metadata={"source": "jsonschemabench", "subset": subset},
    )


def jsonschema_rows_to_samples(
    rows: list[dict[str, Any]],
    subset: str | None = None,
) -> list[EvalSample]:
    return [jsonschema_row_to_sample(row, subset=subset) for row in rows]


# Recommended open structured-output dataset for the generation family.
JSONSCHEMABENCH_HF_ID = "epfl-dlab/JSONSchemaBench"
JSONSCHEMABENCH_DEFAULT_CONFIG = "default"


def load_jsonschemabench_samples(
    config: str = JSONSCHEMABENCH_DEFAULT_CONFIG,
    split: str = "test",
    limit: int = 16,
    cache_path: str | Path | None = None,
) -> list[EvalSample]:
    """Load JSONSchemaBench rows from Hugging Face into generation EvalSamples.

    Uses the optional ``datasets`` dependency and caches raw rows to
    ``cache_path`` (JSONL) for offline reuse.
    """

    if cache_path is not None and Path(cache_path).exists():
        rows = read_jsonl(cache_path)[:limit]
        return jsonschema_rows_to_samples(rows, subset=config)
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without datasets
        raise OptionalDependencyError("datasets", "openvino") from exc
    dataset = load_dataset(JSONSCHEMABENCH_HF_ID, config, split=f"{split}[:{limit}]")
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        row = dict(item)
        row.setdefault("id", f"jsonschemabench_{idx:03d}")
        # Normalize the schema field to a JSON string so it survives JSONL caching.
        if isinstance(row.get("json_schema"), dict):
            row["json_schema"] = json.dumps(row["json_schema"])
        rows.append(row)
    if cache_path is not None:
        write_jsonl(rows, cache_path)
    return jsonschema_rows_to_samples(rows, subset=config)


__all__ = [
    "JSONSCHEMABENCH_HF_ID",
    "jsonschema_row_to_sample",
    "jsonschema_rows_to_samples",
    "load_jsonschemabench_samples",
]
