from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from oviqs.adapters.datasets import JsonlDatasetAdapter
from oviqs.adapters.observability import NullMetricsSink, NullTraceSink
from oviqs.adapters.plugins import InMemoryPluginRegistry
from oviqs.adapters.reporting import (
    CanonicalReportWriter,
    JsonReportAdapter,
    MarkdownReportRenderer,
    ReferenceComparisonAdapter,
)
from oviqs.adapters.runners.factory import build_generation_runner, build_logits_runner
from oviqs.adapters.storage import LocalArtifactStorage
from oviqs.ports.datasets import DatasetReaderPort, DatasetRowsReaderPort
from oviqs.ports.reporting import (
    EvaluationReportWriterPort,
    ReferenceComparisonWriterPort,
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
    report_io: ReportIOPort = field(default_factory=JsonReportAdapter)
    markdown_renderer: ReportRendererPort = field(default_factory=MarkdownReportRenderer)
    reference_comparison_writer: ReferenceComparisonWriterPort = field(
        default_factory=ReferenceComparisonAdapter
    )
    storage: LocalArtifactStorage = field(default_factory=LocalArtifactStorage)
    plugins: InMemoryPluginRegistry = field(default_factory=InMemoryPluginRegistry)
    metrics_sink: NullMetricsSink = field(default_factory=NullMetricsSink)
    trace_sink: NullTraceSink = field(default_factory=NullTraceSink)
    settings: dict[str, Any] = field(default_factory=dict)


def build_default_container(settings: dict[str, Any] | None = None) -> BootstrapContainer:
    return BootstrapContainer(settings=dict(settings or {}))
