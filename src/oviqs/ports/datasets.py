from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from oviqs.domain.samples import EvalSample


class DatasetReaderPort(Protocol):
    def read_samples(self, path: Path) -> list[EvalSample]: ...


class DatasetWriterPort(Protocol):
    def write_samples(self, samples: Iterable[EvalSample], path: Path) -> None: ...


class DatasetRowsReaderPort(Protocol):
    def read_rows(self, path: Path) -> list[dict[str, Any]]: ...
