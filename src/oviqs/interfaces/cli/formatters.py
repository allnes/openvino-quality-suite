from __future__ import annotations

from pathlib import Path


def wrote_message(path: str | Path) -> str:
    return f"Wrote {path}"


__all__ = ["wrote_message"]
