from __future__ import annotations

from pathlib import Path


class PathPolicy:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve_for_read(self, path: str | Path) -> Path:
        return self._resolve_inside_root(path)

    def resolve_for_write(self, path: str | Path) -> Path:
        resolved = self._resolve_inside_root(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def _resolve_inside_root(self, path: str | Path) -> Path:
        resolved = Path(path).expanduser().resolve()
        if resolved == self.root or self.root in resolved.parents:
            return resolved
        raise ValueError(f"Path escapes configured root: {path}")
