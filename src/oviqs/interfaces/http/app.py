from __future__ import annotations

from importlib import import_module
from typing import Any

from oviqs.application.dto.requests import GpuSuiteRequest, LikelihoodEvaluationRequest
from oviqs.application.services.evaluations import build_gpu_suite_report, build_likelihood_report
from oviqs.interfaces.http.schemas import (
    GpuSuiteRunRequest,
    HealthResponse,
    LikelihoodRunRequest,
    ReportResponse,
)
from oviqs.platform.bootstrap import build_default_container


def create_app() -> Any:
    try:
        fastapi = import_module("fastapi")
    except ImportError as exc:
        raise RuntimeError(
            "HTTP interface requires the optional FastAPI dependency. "
            "Install FastAPI in the runtime environment before calling create_app()."
        ) from exc

    app = fastapi.FastAPI(title="OVIQS", version="0.1.0")
    container = build_default_container()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/runs/likelihood", response_model=ReportResponse)
    def run_likelihood(request: LikelihoodRunRequest) -> ReportResponse:
        report = build_likelihood_report(
            LikelihoodEvaluationRequest(**request.model_dump()),
            container.runner_factory,
            container.dataset_reader,
        )
        container.report_writer.write(report, request.out)
        return _response(report.model_dump(mode="json"))

    @app.post("/runs/gpu-suite", response_model=ReportResponse)
    def run_gpu_suite(request: GpuSuiteRunRequest) -> ReportResponse:
        report = build_gpu_suite_report(
            GpuSuiteRequest(**request.model_dump()),
            container.runner_factory,
            container.generation_runner_factory,
            container.dataset_reader,
        )
        container.report_writer.write(report, request.out)
        return _response(report.model_dump(mode="json"))

    return app


def _response(report: dict[str, Any]) -> ReportResponse:
    return ReportResponse(
        run_id=str(report.get("run", {}).get("id", "")),
        overall_status=str(report.get("summary", {}).get("overall_status", "unknown")),
        report=report,
    )
