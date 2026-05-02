from __future__ import annotations

from oviqs.interfaces.http.routes_datasets import build_datasets_router
from oviqs.interfaces.http.routes_models import build_models_router
from oviqs.interfaces.http.routes_runs import build_runs_router
from oviqs.interfaces.http.schemas import (
    GpuSuiteRunRequest,
    HealthResponse,
    LikelihoodRunRequest,
    ReportResponse,
)

__all__ = [
    "GpuSuiteRunRequest",
    "HealthResponse",
    "LikelihoodRunRequest",
    "ReportResponse",
    "build_datasets_router",
    "build_models_router",
    "build_runs_router",
]
