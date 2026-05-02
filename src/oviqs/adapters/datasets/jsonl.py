from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from oviqs.domain.samples import EvalSample


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl_samples(path: str | Path) -> list[EvalSample]:
    return [EvalSample.model_validate(row) for row in read_jsonl(path)]


class JsonlDatasetAdapter:
    def read_rows(self, path: Path) -> list[dict[str, Any]]:
        return read_jsonl(path)

    def read_samples(self, path: Path) -> list[EvalSample]:
        return load_jsonl_samples(path)

    def write_samples(self, samples: Iterable[EvalSample], path: Path) -> None:
        write_jsonl((sample.model_dump(mode="json") for sample in samples), path)
