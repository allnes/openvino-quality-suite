from __future__ import annotations

from pathlib import Path
from typing import Any

from oviqs.application.dto.requests import GpuSuiteRequest


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


def evaluation_response_mapping(report: dict[str, Any], report_uri: str = "") -> dict[str, Any]:
    return {
        "run_id": str(report.get("run", {}).get("id", "")),
        "overall_status": str(report.get("summary", {}).get("overall_status", "unknown")),
        "metrics": _flatten_metrics(report),
        "report_uri": report_uri,
    }


def _flatten_metrics(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for section_name, section in report.items():
        if not isinstance(section, dict) or section_name in {"run", "summary"}:
            continue
        for metric_name, value in section.items():
            if isinstance(value, int | float):
                metrics.append(
                    {
                        "name": f"{section_name}.{metric_name}",
                        "value": float(value),
                        "status": str(section.get("status", "unknown")),
                        "reference": "",
                    }
                )
    return metrics
