from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from oviqs.ports.runners import RunnerCapabilities


class BaseRunner(ABC):
    name: str = "runner"
    capabilities = RunnerCapabilities()

    @abstractmethod
    def run_info(self) -> dict[str, Any]:
        raise NotImplementedError

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        return None


class BaseLogitsRunner(BaseRunner):
    @abstractmethod
    def forward_logits(self, input_ids: Any, attention_mask: Any | None = None) -> Any:
        """Return logits with shape [B, T, V]."""
        raise NotImplementedError


class BaseGenerationRunner(BaseRunner):
    @abstractmethod
    def generate(self, prompt: str, **generation_kwargs: Any) -> Any:
        raise NotImplementedError


__all__ = ["BaseGenerationRunner", "BaseLogitsRunner", "BaseRunner"]
