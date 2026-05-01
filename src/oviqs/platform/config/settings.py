from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Settings(BaseModel):
    profile: str = "base"
    defaults: dict[str, Any] = Field(default_factory=dict)
    quality_gates: dict[str, Any] = Field(default_factory=dict)


def load_settings(path: Path) -> Settings:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings.model_validate(payload)
