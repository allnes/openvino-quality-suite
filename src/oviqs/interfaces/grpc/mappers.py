from __future__ import annotations

from pathlib import Path
from typing import Any

from oviqs.application.dto.requests import GpuSuiteRequest
from oviqs.application.reporting import ReportAnalysisService, build_report_interface_response


def gpu_suite_request_from_mapping(payload: dict[str, Any]) -> GpuSuiteRequest:
    return GpuSuiteRequest(
        model=str(payload["model"]),
        dataset=Path(str(payload["dataset_uri"])),
        out=Path(str(payload["report_uri"])),
        backend=str(payload.get("backend") or "openvino-runtime"),
        device=str(payload.get("device") or "GPU"),
        window_size=int(payload.get("window_size") or 64),
        stride=int(payload.get("stride") or 32),
        genai_model=payload.get("genai_model"),
    )


def evaluation_response_mapping(
    report: dict[str, Any],
    report_uri: str = "",
    analysis_service: ReportAnalysisService | None = None,
) -> dict[str, Any]:
    return build_report_interface_response(
        report,
        analysis_service=analysis_service,
        report_uri=report_uri,
    ).grpc_mapping()
