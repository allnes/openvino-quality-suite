from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from oviqs.core.sample import EvalSample
from oviqs.datasets.jsonl import load_jsonl_samples, read_jsonl, write_jsonl


class JsonlDatasetAdapter:
    def read_rows(self, path: Path) -> list[dict[str, Any]]:
        return read_jsonl(path)

    def read_samples(self, path: Path) -> list[EvalSample]:
        return load_jsonl_samples(path)

    def write_samples(self, samples: Iterable[EvalSample], path: Path) -> None:
        write_jsonl((sample.model_dump(mode="json") for sample in samples), path)
