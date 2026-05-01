from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from oviqs.platform.config.settings import Settings


def load_profile_settings(
    base_path: Path,
    profile_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    payload = _read_yaml(base_path)
    if profile_path is not None:
        payload = _deep_merge(payload, _read_yaml(profile_path))
    if overrides:
        payload = _deep_merge(payload, dict(overrides))
    return Settings.model_validate(payload)


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged
