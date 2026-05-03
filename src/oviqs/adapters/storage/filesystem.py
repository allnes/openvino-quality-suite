from __future__ import annotations

from pathlib import Path

from oviqs.platform.security import PathPolicy


class LocalArtifactStorage:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).expanduser().resolve() if root is not None else None
        self.policy = PathPolicy(self.root) if self.root is not None else None

    def ensure_dir(self, path: str | Path) -> None:
        self._resolve_path(path).mkdir(parents=True, exist_ok=True)

    def read_text(self, path: str | Path) -> str:
        return self._resolve_path(path).read_text(encoding="utf-8")

    def write_text(self, path: str | Path, content: str) -> None:
        target = self._resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def exists(self, path: str | Path) -> bool:
        return self._resolve_path(path).exists()

    def _resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if self.root is not None and not candidate.is_absolute():
            candidate = self.root / candidate
        if self.policy is None:
            return candidate.resolve()
        return self.policy.resolve_for_read(candidate)


__all__ = ["LocalArtifactStorage"]
