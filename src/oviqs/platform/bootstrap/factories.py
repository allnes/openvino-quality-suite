from __future__ import annotations

from oviqs.adapters.plugins import EntryPointPluginRegistry
from oviqs.adapters.runners.factory import build_generation_runner, build_logits_runner
from oviqs.ports.analysis import AnalysisRulePort, GateEvaluatorPort, MetricCatalogPort
from oviqs.ports.artifacts import MetricTableWriterPort, SampleMetricsStorePort
from oviqs.ports.reporting import (
    ReferenceComparisonRendererPort,
    ReportIOPort,
    ReportRendererPort,
)


def build_report_io() -> ReportIOPort:
    return _plugin_instance("oviqs.reporters", "json")


def build_markdown_renderer() -> ReportRendererPort:
    return _plugin_instance("oviqs.reporters", "markdown")


def build_html_renderer() -> ReportRendererPort:
    return _plugin_instance("oviqs.reporters", "html-dashboard")


def build_csv_metrics_writer() -> MetricTableWriterPort:
    return _plugin_instance("oviqs.reporters", "metrics-csv")


def build_sample_metrics_store() -> SampleMetricsStorePort:
    return _plugin_instance("oviqs.reporters", "sample-jsonl")


def build_reference_comparison_renderer() -> ReferenceComparisonRendererPort:
    return _plugin_instance("oviqs.reporters", "reference-comparison")


def build_analysis_rules() -> tuple[AnalysisRulePort, ...]:
    return (_plugin_instance("oviqs.analysis_rules", "built-in"),)


def build_metric_catalog() -> MetricCatalogPort:
    return _plugin_instance("oviqs.metric_catalogs", "default")


def build_gate_evaluator() -> GateEvaluatorPort:
    return _plugin_instance("oviqs.gate_evaluators", "default")


def _plugin_instance(group: str, name: str):
    plugin = EntryPointPluginRegistry(group).get(name)
    return plugin()


__all__ = [
    "build_analysis_rules",
    "build_csv_metrics_writer",
    "build_gate_evaluator",
    "build_generation_runner",
    "build_html_renderer",
    "build_logits_runner",
    "build_markdown_renderer",
    "build_metric_catalog",
    "build_reference_comparison_renderer",
    "build_report_io",
    "build_sample_metrics_store",
]
