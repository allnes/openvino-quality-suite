from __future__ import annotations

import os


def read_secret(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def require_secret(name: str) -> str:
    value = read_secret(name)
    if value is None or value == "":
        raise RuntimeError(f"Required secret is not configured: {name}")
    return value


__all__ = ["read_secret", "require_secret"]
