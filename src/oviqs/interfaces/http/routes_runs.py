from __future__ import annotations

from typing import Any

from oviqs.application.dto.requests import GpuSuiteRequest, LikelihoodEvaluationRequest
from oviqs.application.reporting import build_report_interface_response
from oviqs.application.services.evaluations import build_gpu_suite_report, build_likelihood_report
from oviqs.interfaces.http.schemas import (
    GpuSuiteRunRequest,
    HealthResponse,
    LikelihoodRunRequest,
    ReportResponse,
)


def build_runs_router(fastapi: Any, container: Any) -> Any:
    router = fastapi.APIRouter()

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @router.post("/v1/runs/likelihood", response_model=ReportResponse)
    @router.post("/runs/likelihood", response_model=ReportResponse, include_in_schema=False)
    def run_likelihood(request: LikelihoodRunRequest) -> ReportResponse:
        report = build_likelihood_report(
            LikelihoodEvaluationRequest(**request.model_dump()),
            container.runner_factory,
            container.dataset_reader,
        )
        container.report_writer.write(report, request.out)
        return ReportResponse(
            **build_report_interface_response(
                report.model_dump(mode="json"),
                analysis_service=container.report_analysis_service(),
                report_uri=str(request.out),
            ).http_payload()
        )

    @router.post("/v1/runs/gpu-suite", response_model=ReportResponse)
    @router.post("/runs/gpu-suite", response_model=ReportResponse, include_in_schema=False)
    def run_gpu_suite(request: GpuSuiteRunRequest) -> ReportResponse:
        report = build_gpu_suite_report(
            GpuSuiteRequest(**request.model_dump()),
            container.runner_factory,
            container.generation_runner_factory,
            container.dataset_reader,
        )
        container.report_writer.write(report, request.out)
        return ReportResponse(
            **build_report_interface_response(
                report.model_dump(mode="json"),
                analysis_service=container.report_analysis_service(),
                report_uri=str(request.out),
            ).http_payload()
        )

    return router


__all__ = ["build_runs_router"]
