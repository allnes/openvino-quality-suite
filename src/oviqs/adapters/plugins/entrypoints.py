from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]


class EntryPointPluginRegistry:
    def __init__(self, group: str) -> None:
        self.group = group

    def names(self) -> Iterable[str]:
        return [entry_point.name for entry_point in self._entry_points()]

    def get(self, name: str) -> Any:
        for entry_point in self._entry_points():
            if entry_point.name == name:
                return entry_point.load()
        raise KeyError(name)

    def _entry_points(self) -> tuple[EntryPoint | _SourceTreeEntryPoint, ...]:
        discovered: list[EntryPoint | _SourceTreeEntryPoint] = list(
            entry_points().select(group=self.group)
        )
        discovered_names = {entry_point.name for entry_point in discovered}
        for entry_point in _source_tree_entry_points(self.group):
            if entry_point.name not in discovered_names:
                discovered.append(entry_point)
        return tuple(discovered)


@dataclass(frozen=True)
class _SourceTreeEntryPoint:
    name: str
    value: str
    group: str

    def load(self) -> Any:
        module_name, _, attribute_path = self.value.partition(":")
        target: Any = import_module(module_name)
        for part in attribute_path.split("."):
            target = getattr(target, part)
        return target


@lru_cache
def _source_tree_entry_points(group: str) -> tuple[_SourceTreeEntryPoint, ...]:
    pyproject = _find_source_tree_pyproject()
    if pyproject is None:
        return ()
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    entry_points_data = data.get("project", {}).get("entry-points", {}).get(group, {})
    if not isinstance(entry_points_data, dict):
        return ()
    return tuple(
        _SourceTreeEntryPoint(name=name, value=value, group=group)
        for name, value in entry_points_data.items()
        if isinstance(name, str) and isinstance(value, str)
    )


@lru_cache
def _find_source_tree_pyproject() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pyproject.toml"
        source_root = parent / "src" / "oviqs"
        if candidate.exists() and source_root.exists():
            return candidate
    return None
