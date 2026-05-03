from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_gate_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Gate file must contain a mapping: {path}")
    return payload


__all__ = ["load_gate_payload"]
