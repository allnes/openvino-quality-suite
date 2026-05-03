from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "oviqs"


class LikelihoodRunRequest(BaseModel):
    model: str
    dataset: Path
    out: Path
    backend: str = "dummy"
    device: str = "CPU"
    window_size: int = Field(default=4096, gt=0)
    stride: int = Field(default=1024, gt=0)


class GpuSuiteRunRequest(BaseModel):
    model: str
    dataset: Path
    out: Path
    backend: str = "openvino-runtime"
    device: str = "GPU"
    window_size: int = Field(default=64, gt=0)
    stride: int = Field(default=32, gt=0)
    genai_model: str | None = None


class ReportResponse(BaseModel):
    run_id: str
    overall_status: str
    report: dict[str, Any]
    analysis: dict[str, Any]
    metrics: list[dict[str, Any]]
