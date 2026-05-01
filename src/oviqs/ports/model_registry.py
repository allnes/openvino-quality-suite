from __future__ import annotations

from typing import Any, Protocol


class ModelRegistryPort(Protocol):
    def resolve(self, model_id: str) -> dict[str, Any]: ...
