from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oviqs.adapters.analysis import LocalTrendStore
from oviqs.adapters.datasets import JsonlDatasetAdapter
from oviqs.adapters.observability import NullMetricsSink, NullTraceSink
from oviqs.adapters.plugins import InMemoryPluginRegistry
from oviqs.adapters.reporting import CanonicalReportWriter, ReportPackager
from oviqs.adapters.runners.factory import build_generation_runner, build_logits_runner
from oviqs.adapters.storage import LocalArtifactStorage
from oviqs.application.reporting import (
    ReportAnalysisService,
    ReportGenerationService,
    ReportPackageService,
    ReportWorkflowService,
)
from oviqs.platform.bootstrap.factories import (
    build_analysis_rules,
    build_csv_metrics_writer,
    build_gate_evaluator,
    build_html_renderer,
    build_markdown_renderer,
    build_metric_catalog,
    build_reference_comparison_renderer,
    build_report_io,
    build_sample_metrics_store,
)
from oviqs.ports.analysis import (
    AnalysisRulePort,
    GateEvaluatorPort,
    MetricCatalogPort,
    TrendStorePort,
)
from oviqs.ports.artifacts import ArtifactStorePort, MetricTableWriterPort, SampleMetricsStorePort
from oviqs.ports.datasets import DatasetReaderPort, DatasetRowsReaderPort
from oviqs.ports.reporting import (
    EvaluationReportWriterPort,
    ReferenceComparisonRendererPort,
    ReportBundleWriterPort,
    ReportIOPort,
    ReportRendererPort,
)
from oviqs.ports.runners import GenerationRunnerFactoryPort, LogitsRunnerFactoryPort


@dataclass(frozen=True)
class BootstrapContainer:
    runner_factory: LogitsRunnerFactoryPort = build_logits_runner
    generation_runner_factory: GenerationRunnerFactoryPort = build_generation_runner
    dataset_reader: DatasetReaderPort = field(default_factory=JsonlDatasetAdapter)
    rows_reader: DatasetRowsReaderPort = field(default_factory=JsonlDatasetAdapter)
    report_writer: EvaluationReportWriterPort = field(default_factory=CanonicalReportWriter)
    report_io: ReportIOPort = field(default_factory=build_report_io)
    markdown_renderer: ReportRendererPort = field(default_factory=build_markdown_renderer)
    html_renderer: ReportRendererPort = field(default_factory=build_html_renderer)
    csv_metrics_writer: MetricTableWriterPort = field(default_factory=build_csv_metrics_writer)
    sample_metrics_store: SampleMetricsStorePort = field(default_factory=build_sample_metrics_store)
    analysis_rules: tuple[AnalysisRulePort, ...] = field(default_factory=build_analysis_rules)
    metric_catalog: MetricCatalogPort = field(default_factory=build_metric_catalog)
    gate_evaluator: GateEvaluatorPort = field(default_factory=build_gate_evaluator)
    reference_comparison_renderer: ReferenceComparisonRendererPort = field(
        default_factory=build_reference_comparison_renderer
    )
    storage: ArtifactStorePort = field(default_factory=LocalArtifactStorage)
    plugins: InMemoryPluginRegistry = field(default_factory=InMemoryPluginRegistry)
    metrics_sink: NullMetricsSink = field(default_factory=NullMetricsSink)
    trace_sink: NullTraceSink = field(default_factory=NullTraceSink)
    settings: dict[str, Any] = field(default_factory=dict)

    def report_analysis_service(
        self,
        trend_store: TrendStorePort | None = None,
    ) -> ReportAnalysisService:
        return ReportAnalysisService(
            rules=self.analysis_rules,
            metric_catalog=self.metric_catalog,
            trend_store=trend_store,
        )

    def report_generation_service(self) -> ReportGenerationService:
        return ReportGenerationService(self.report_analysis_service())

    def report_package_service(self) -> ReportPackageService:
        return ReportPackageService(
            self.markdown_renderer,
            self.html_renderer,
            self.csv_metrics_writer,
            self.storage,
            self.sample_metrics_store,
            generation_service=self.report_generation_service(),
        )

    def report_bundle_writer(self) -> ReportBundleWriterPort:
        return ReportPackager(self.report_package_service())

    def report_workflow_service(
        self,
        trend_history_path: Path | None = None,
    ) -> ReportWorkflowService:
        trend_store = (
            LocalTrendStore(history_path=trend_history_path) if trend_history_path else None
        )
        return ReportWorkflowService(
            report_io=self.report_io,
            artifact_store=self.storage,
            analysis_service=self.report_analysis_service(trend_store=trend_store),
            gate_evaluator=self.gate_evaluator,
            bundle_writer=self.report_bundle_writer(),
            markdown_renderer=self.markdown_renderer,
            html_renderer=self.html_renderer,
            reference_comparison_renderer=self.reference_comparison_renderer,
            metrics_writer=self.csv_metrics_writer,
        )


def build_default_container(settings: dict[str, Any] | None = None) -> BootstrapContainer:
    return BootstrapContainer(settings=dict(settings or {}))
