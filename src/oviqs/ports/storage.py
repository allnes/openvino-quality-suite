from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ArtifactStoragePort(Protocol):
    def read_text(self, _uri: str | Path) -> str: ...

    def write_text(self, _uri: str | Path, content: str) -> None: ...

    def exists(self, _uri: str | Path) -> bool: ...
