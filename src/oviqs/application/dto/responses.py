from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunResponse:
    run_id: str
    overall_status: str = "unknown"
    report: dict[str, Any] = field(default_factory=dict)


__all__ = ["RunResponse"]
