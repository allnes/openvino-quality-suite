from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oviqs.application.reporting.generation_service import ReportGenerationService
from oviqs.application.reporting.validation import (
    validate_report_bundle_metadata,
    validate_sample_metrics_contract,
)
from oviqs.domain.reporting import ReportAnalysis, ReportBundle
from oviqs.domain.reporting.view_model import build_report_view_model
from oviqs.ports.artifacts import ArtifactStorePort, MetricTableWriterPort, SampleMetricsStorePort
from oviqs.ports.reporting import ReportRendererPort


class ReportPackageService:
    def __init__(
        self,
        markdown_renderer: ReportRendererPort,
        html_renderer: ReportRendererPort,
        csv_writer: MetricTableWriterPort,
        artifact_store: ArtifactStorePort,
        sample_metrics_store: SampleMetricsStorePort,
        generation_service: ReportGenerationService,
    ) -> None:
        self.generation_service = generation_service
        self.markdown_renderer = markdown_renderer
        self.html_renderer = html_renderer
        self.csv_writer = csv_writer
        self.artifact_store = artifact_store
        self.sample_metrics_store = sample_metrics_store

    def build(
        self,
        report: dict[str, Any],
        out: Path,
        *,
        baseline: dict[str, Any] | None = None,
        gates: dict[str, Any] | None = None,
        analysis: ReportAnalysis | None = None,
    ) -> ReportBundle:
        self.artifact_store.ensure_dir(out)
        generated = self.generation_service.enrich(
            report,
            baseline=baseline,
            gates=gates,
            analysis=analysis,
        )
        analysis = generated.analysis
        report_payload = generated.report
        view_model = build_report_view_model(report_payload, analysis)
        report_payload.setdefault("report_metadata", {})["bundle_root"] = str(out)
        report_payload.setdefault("sample_metrics_summary", {})["outliers"] = (
            analysis.to_dict().get("sample_outliers", [])
        )

        report_json = out / "report.json"
        analysis_json = out / "analysis.json"
        metrics_csv = out / "metrics.csv"
        sample_metrics_jsonl = out / "sample_metrics.jsonl"
        index_md = out / "index.md"
        dashboard_html = out / "dashboard.html"
        assets_dir = out / "assets"
        metadata_json = out / "metadata.json"
        self.artifact_store.ensure_dir(assets_dir)

        self.artifact_store.write_text(
            report_json,
            json.dumps(report_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
        self.artifact_store.write_text(
            analysis_json,
            json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
        self.artifact_store.write_text(metrics_csv, self.csv_writer.render(list(analysis.metrics)))
        sample_metrics = list(_iter_sample_metrics(report_payload))
        self.artifact_store.write_text(
            sample_metrics_jsonl,
            self.sample_metrics_store.render_metrics(sample_metrics),
        )
        self.artifact_store.write_text(index_md, self.markdown_renderer.render(view_model))
        self.artifact_store.write_text(dashboard_html, self.html_renderer.render(view_model))

        bundle = ReportBundle(
            root=str(out),
            report_json=str(report_json),
            analysis_json=str(analysis_json),
            metrics_csv=str(metrics_csv),
            sample_metrics_jsonl=str(sample_metrics_jsonl),
            index_md=str(index_md),
            dashboard_html=str(dashboard_html),
            metadata_json=str(metadata_json),
        )
        metadata = bundle.to_dict()
        metadata["assets_dir"] = str(assets_dir)
        metadata["validation_errors"] = validate_report_bundle_metadata(metadata)
        metadata["sample_metrics_validation_errors"] = validate_sample_metrics_contract(
            sample_metrics
        )
        self.artifact_store.write_text(
            metadata_json,
            json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
        return bundle


def _iter_sample_metrics(report: dict[str, Any]):
    for section, payload in report.items():
        if not isinstance(payload, dict):
            continue
        samples = payload.get("samples")
        if not isinstance(samples, list):
            continue
        for idx, sample in enumerate(samples):
            if isinstance(sample, dict):
                yield {"section": section, "sample_index": idx, **sample}
            else:
                yield {"section": section, "sample_index": idx, "value": sample}


__all__ = ["ReportPackageService"]
