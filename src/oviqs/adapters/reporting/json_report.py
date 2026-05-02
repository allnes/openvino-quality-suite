from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonReportAdapter:
    def load(self, path: Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def write(self, report: dict[str, Any], path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )


__all__ = ["JsonReportAdapter"]
