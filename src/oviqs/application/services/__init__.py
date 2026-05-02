from __future__ import annotations

from oviqs.application.services.compare_reports import compare_reports
from oviqs.application.services.evaluate_agent import build_agent_report
from oviqs.application.services.evaluate_drift import build_drift_report
from oviqs.application.services.evaluate_gpu_suite import build_gpu_suite_report
from oviqs.application.services.evaluate_likelihood import build_likelihood_report
from oviqs.application.services.evaluate_long_context import build_long_context_report
from oviqs.application.services.evaluate_rag import build_rag_report
from oviqs.application.services.evaluate_serving import build_serving_report
from oviqs.application.services.render_report import (
    render_report_to_path,
    write_reference_comparison_table,
)
from oviqs.application.services.run_suite import (
    build_suite_scaffold_report,
    write_suite_scaffold_report,
)

__all__ = [
    "build_suite_scaffold_report",
    "build_agent_report",
    "build_drift_report",
    "build_gpu_suite_report",
    "build_likelihood_report",
    "build_long_context_report",
    "build_rag_report",
    "build_serving_report",
    "compare_reports",
    "render_report_to_path",
    "write_reference_comparison_table",
    "write_suite_scaffold_report",
]
