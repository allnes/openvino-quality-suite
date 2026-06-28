from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LikelihoodEvaluationRequest:
    model: str
    dataset: Path
    out: Path
    backend: str = "dummy"
    device: str = "CPU"
    window_size: int = 4096
    stride: int = 1024


@dataclass(frozen=True)
class DriftEvaluationRequest:
    reference: str
    current: str
    dataset: Path
    out: Path
    reference_backend: str = "dummy"
    current_backend: str = "dummy"
    reference_device: str = "GPU"
    device: str = "GPU"


@dataclass(frozen=True)
class LongContextEvaluationRequest:
    dataset: Path
    out: Path
    model: str = "dummy"
    backend: str = "dummy"
    device: str = "CPU"
    lengths: tuple[int, ...] = (4096, 8192, 16384)
    window_size: int = 4096
    stride: int = 1024


@dataclass(frozen=True)
class ServingEvaluationRequest:
    out: Path
    model: str = "dummy"
    dataset: Path | None = None
    backend: str = "dummy"
    device: str = "CPU"


@dataclass(frozen=True)
class RagEvaluationRequest:
    dataset: Path
    out: Path
    answers: Path | None = None
    scorer: str = "placeholder"


@dataclass(frozen=True)
class AgentEvaluationRequest:
    traces: Path
    out: Path
    expected: Path | None = None


@dataclass(frozen=True)
class GpuSuiteRequest:
    model: str
    dataset: Path
    out: Path
    backend: str = "openvino-runtime"
    device: str = "GPU"
    window_size: int = 64
    stride: int = 32
    genai_model: str | None = None
    # PyTorch/HF reference for inference equivalence (OpenVINO vs PyTorch drift),
    # both on the Intel GPU. When set, drift compares the OpenVINO Intel-GPU model
    # against this PyTorch Intel-GPU reference instead of a degenerate self-drift.
    reference_model: str | None = None
    reference_backend: str = "hf"
    reference_device: str = "GPU"
