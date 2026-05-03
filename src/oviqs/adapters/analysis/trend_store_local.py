from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LocalTrendStore:
    def __init__(
        self,
        baseline_path: Path | None = None,
        history_path: Path | None = None,
    ) -> None:
        self.baseline_path = baseline_path
        self.history_path = history_path

    def load_baseline(self, report: dict[str, Any]) -> dict[str, Any] | None:
        if self.baseline_path is not None and self.baseline_path.exists():
            return json.loads(self.baseline_path.read_text(encoding="utf-8"))
        history = self.load_history(report)
        if history:
            return history[-1]
        return None

    def load_history(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        if self.history_path is None or not self.history_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    def append(self, report: dict[str, Any]) -> None:
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")


__all__ = ["LocalTrendStore"]
