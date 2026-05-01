from __future__ import annotations

import json
import logging
from typing import Any


class JsonLogTraceSink:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("oviqs")

    def event(self, name: str, _attributes: dict[str, Any] | None = None) -> None:
        payload = {"event": name, "attributes": _attributes or {}}
        self._logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))
