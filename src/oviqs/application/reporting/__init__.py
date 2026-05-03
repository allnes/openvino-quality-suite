from __future__ import annotations

from oviqs.application.reporting.analysis_service import ReportAnalysisService
from oviqs.application.reporting.comparison_service import ReportComparisonService
from oviqs.application.reporting.gates import load_gate_payload
from oviqs.application.reporting.generation_service import GeneratedReport, ReportGenerationService
from oviqs.application.reporting.interface_response import (
    ReportInterfaceResponse,
    build_report_interface_response,
)
from oviqs.application.reporting.normalization import flatten_report_metrics
from oviqs.application.reporting.package_service import ReportPackageService
from oviqs.application.reporting.reference_comparison_service import ReferenceComparisonService
from oviqs.application.reporting.schema_normalization import (
    UnsupportedReportSchemaVersion,
    normalize_evaluation_report_contract,
)
from oviqs.application.reporting.validation import (
    validate_evaluation_report_contract,
    validate_metric_observation_contract,
    validate_report_bundle_metadata,
    validate_sample_metric_contract,
    validate_sample_metrics_contract,
)
from oviqs.application.reporting.workflow_service import ReportWorkflowService

__all__ = [
    "GeneratedReport",
    "ReportAnalysisService",
    "ReportComparisonService",
    "ReportGenerationService",
    "ReportInterfaceResponse",
    "ReportPackageService",
    "ReportWorkflowService",
    "ReferenceComparisonService",
    "UnsupportedReportSchemaVersion",
    "build_report_interface_response",
    "flatten_report_metrics",
    "load_gate_payload",
    "normalize_evaluation_report_contract",
    "validate_evaluation_report_contract",
    "validate_metric_observation_contract",
    "validate_report_bundle_metadata",
    "validate_sample_metric_contract",
    "validate_sample_metrics_contract",
]
