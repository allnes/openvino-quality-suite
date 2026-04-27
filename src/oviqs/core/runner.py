from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseRunner(ABC):
    name: str = "runner"

    @abstractmethod
    def run_info(self) -> dict[str, Any]:
        raise NotImplementedError


class BaseLogitsRunner(BaseRunner):
    @abstractmethod
    def forward_logits(self, input_ids, attention_mask=None):
        """Return logits with shape [B, T, V]."""
        raise NotImplementedError


class BaseGenerationRunner(BaseRunner):
    @abstractmethod
    def generate(self, prompt: str, **generation_kwargs):
        raise NotImplementedError
