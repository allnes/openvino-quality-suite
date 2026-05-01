from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RunnerCapabilities:
    exposes_logits: bool = False
    supports_generation: bool = False
    supports_stateful_kv: bool = False
    supports_dynamic_shapes: bool = False
    supports_continuous_batching: bool = False
    supported_devices: tuple[str, ...] = field(default_factory=tuple)


class RunnerPort(Protocol):
    name: str

    def run_info(self) -> dict[str, Any]: ...


class LogitsRunnerPort(RunnerPort, Protocol):
    capabilities: RunnerCapabilities

    def warmup(self) -> None: ...

    def forward_logits(self, input_ids: Any, attention_mask: Any | None = None) -> Any: ...

    def close(self) -> None: ...


class GenerationRunnerPort(RunnerPort, Protocol):
    capabilities: RunnerCapabilities

    def generate(self, prompt: str, **generation_kwargs: Any) -> Any: ...

    def close(self) -> None: ...


class LogitsRunnerFactoryPort(Protocol):
    def __call__(self, backend: str, model: str, device: str) -> LogitsRunnerPort: ...


class GenerationRunnerFactoryPort(Protocol):
    def __call__(self, backend: str, model: str, device: str) -> GenerationRunnerPort: ...
