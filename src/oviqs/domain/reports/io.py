from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _default(value: Any) -> Any:
    """JSON encoder fallback for pydantic models and numpy scalars/arrays."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    # numpy scalars/arrays without importing numpy as a hard dependency
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def report_to_jsonable(report: Any) -> Any:
    """Normalize a report (pydantic model or plain dict/list) to JSON-ready data."""
    if isinstance(report, BaseModel):
        return report.model_dump(mode="json")
    return report


def write_report(report: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Write an evaluation report to ``path`` as JSON.

    Accepts either a pydantic report model (e.g. ``EvaluationReport``) or a plain
    dict assembled by the metric scripts, and tolerates numpy scalar/array values
    that the matrix/extended scripts emit.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = report_to_jsonable(report)
    out.write_text(
        json.dumps(payload, indent=indent, sort_keys=True, default=_default),
        encoding="utf-8",
    )
    return out


__all__ = ["report_to_jsonable", "write_report"]
