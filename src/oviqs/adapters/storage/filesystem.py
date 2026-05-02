from __future__ import annotations

from pathlib import Path


class LocalArtifactStorage:
    def read_text(self, uri: str | Path) -> str:
        return Path(uri).read_text(encoding="utf-8")

    def write_text(self, uri: str | Path, content: str) -> None:
        path = Path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def exists(self, uri: str | Path) -> bool:
        return Path(uri).exists()


__all__ = ["LocalArtifactStorage"]
