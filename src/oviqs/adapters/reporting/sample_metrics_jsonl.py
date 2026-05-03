from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlSampleMetricsStore:
    def render_metrics(self, metrics: list[dict[str, Any]]) -> str:
        return "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in metrics
        )

    def write_metrics(self, metrics: list[dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render_metrics(metrics), encoding="utf-8")


__all__ = ["JsonlSampleMetricsStore"]
