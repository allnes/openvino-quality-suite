from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oviqs.application.reporting.analysis_service import ReportAnalysisService
from oviqs.application.reporting.gates import load_gate_payload
from oviqs.application.reporting.reference_comparison_service import ReferenceComparisonService
from oviqs.application.reporting.validation import validate_evaluation_report_contract
from oviqs.domain.reporting import ReportBundle, SuiteBundle, build_suite_model_summary
from oviqs.domain.reporting.view_model import build_report_view_model
from oviqs.ports.analysis import GateEvaluatorPort
from oviqs.ports.artifacts import ArtifactStorePort, MetricTableWriterPort
from oviqs.ports.reporting import (
    ReferenceComparisonRendererPort,
    ReportBundleWriterPort,
    ReportIOPort,
    ReportRendererPort,
    SuiteIndexRendererPort,
)


@dataclass(frozen=True)
class ReportWorkflowService:
    report_io: ReportIOPort
    artifact_store: ArtifactStorePort
    analysis_service: ReportAnalysisService
    gate_evaluator: GateEvaluatorPort
    bundle_writer: ReportBundleWriterPort
    markdown_renderer: ReportRendererPort
    html_renderer: ReportRendererPort
    reference_comparison_renderer: ReferenceComparisonRendererPort
    suite_index_renderer: SuiteIndexRendererPort
    metrics_writer: MetricTableWriterPort

    def build_bundle(
        self,
        report_path: Path,
        out: Path,
        *,
        baseline_path: Path | None = None,
        gates_path: Path | None = None,
        format_name: str = "all",
    ) -> ReportBundle:
        if format_name != "all":
            raise ValueError("Only --format all is supported for report build")
        current = self.report_io.load(report_path)
        baseline = self.report_io.load(baseline_path) if baseline_path else None
        gates = self._evaluate_gates(current, gates_path)
        if gates:
            current = {**current, "gates": gates}
        analysis = self.analysis_service.analyze(current, baseline=baseline, gates=gates)
        return self.bundle_writer.write_bundle(current, analysis, out)

    def build_suite_bundle(
        self,
        reports: list[str],
        out: Path,
        *,
        gates_path: Path | None = None,
        title: str = "OVIQS — OpenVINO Inference Quality Suite",
    ) -> SuiteBundle:
        """Build a multi-model suite bundle.

        For each ``LABEL=PATH`` report this writes a per-model bundle under
        ``out/models/<label>/``, a cross-model ``comparison.html``, and an
        ``index.html`` landing page linking them with a fidelity summary table.
        """
        if not reports:
            raise ValueError("At least one report is required for a suite bundle")
        parsed = [_parse_labeled_report_path(value) for value in reports]
        all_labels = [label for label, _ in parsed]
        labels = [_unique_slug(label, all_labels, idx) for idx, (label, _) in enumerate(parsed)]

        models_root = out / "models"
        summaries = []
        bundle_dirs: list[str] = []
        comparison_inputs: list[str] = []
        for slug, (label, path) in zip(labels, parsed, strict=True):
            model_out = models_root / slug
            self.build_bundle(path, model_out, gates_path=gates_path)
            report = self.report_io.load(path)
            rel_dir = f"models/{slug}"
            summaries.append(build_suite_model_summary(label, report, rel_dir))
            bundle_dirs.append(str(model_out))
            comparison_inputs.append(f"{label}={path}")

        comparison_href = None
        comparison_path = out / "comparison.html"
        try:
            self.write_reference_comparison(
                comparison_inputs, comparison_path, format_name="html-by-model"
            )
            comparison_href = "comparison.html"
        except ValueError:
            comparison_path = None  # comparison is best-effort; index still renders

        index_html = out / "index.html"
        self.artifact_store.write_text(
            index_html,
            self.suite_index_renderer.render(
                summaries, comparison_href=comparison_href, title=title
            ),
        )
        return SuiteBundle(
            root=str(out),
            index_html=str(index_html),
            comparison_html=str(comparison_path) if comparison_path else None,
            model_bundles=tuple(bundle_dirs),
        )

    def analyze_report(
        self,
        report_path: Path,
        out: Path,
        *,
        baseline_path: Path | None = None,
        gates_path: Path | None = None,
    ) -> dict[str, Any]:
        current = self.report_io.load(report_path)
        baseline = self.report_io.load(baseline_path) if baseline_path else None
        gates = self._evaluate_gates(current, gates_path)
        if gates:
            current = {**current, "gates": gates}
        analysis = self.analysis_service.analyze(
            current,
            baseline=baseline,
            gates=gates or current.get("gates"),
        ).to_dict()
        self.artifact_store.write_text(
            out,
            json.dumps(analysis, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        )
        return analysis

    def render_bundle(self, bundle: Path, out: Path, *, format_name: str) -> None:
        report_path = bundle / "report.json"
        if not report_path.exists():
            raise ValueError(f"Bundle report not found: {report_path}")
        self.render_report(report_path, out, format_name=format_name)

    def render_report(self, report_path: Path, out: Path, *, format_name: str) -> None:
        renderer = self._renderer(format_name)
        report = self.report_io.load(report_path)
        analysis = self.analysis_service.analyze(report, gates=report.get("gates"))
        report = {**report, "analysis": analysis.to_dict()}
        view_model = build_report_view_model(report, analysis)
        self.artifact_store.write_text(out, renderer.render(view_model))

    def write_metrics_table(self, report_path: Path, out: Path) -> None:
        report = self.report_io.load(report_path)
        analysis = self.analysis_service.analyze(report)
        self.artifact_store.write_text(out, self.metrics_writer.render(list(analysis.metrics)))

    def validate_report(self, report_path: Path) -> list[str]:
        report = self.report_io.load(report_path)
        return validate_evaluation_report_contract(report)

    def write_reference_comparison(
        self,
        reports: list[str],
        out: Path,
        *,
        format_name: str = "markdown",
        include_all_metrics: bool = False,
    ) -> None:
        loaded_reports = []
        for value in reports:
            label, path = _parse_labeled_report_path(value)
            loaded_reports.append((label, self.report_io.load(path), str(path)))
        comparison = ReferenceComparisonService().build(
            loaded_reports,
            include_all_metrics=include_all_metrics,
        )
        self.artifact_store.write_text(
            out,
            self.reference_comparison_renderer.render(comparison, format_name),
        )

    def _evaluate_gates(
        self,
        report: dict[str, Any],
        gates_path: Path | None,
    ) -> dict[str, Any]:
        return self.gate_evaluator.evaluate(report, load_gate_payload(gates_path))

    def _renderer(self, format_name: str) -> ReportRendererPort:
        if format_name == self.markdown_renderer.format_name:
            return self.markdown_renderer
        if format_name == self.html_renderer.format_name:
            return self.html_renderer
        raise ValueError("Supported formats: markdown, html-dashboard")


def _parse_labeled_report_path(value: str) -> tuple[str, Path]:
    label, sep, path = value.partition("=")
    if sep:
        return label, Path(path)
    path_obj = Path(value)
    return path_obj.stem, path_obj


def _slugify(label: str) -> str:
    slug = "".join(ch if ch.isalnum() else "-" for ch in label.lower()).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "model"


def _unique_slug(label: str, all_labels: list[str], index: int) -> str:
    """Filesystem-safe, collision-free directory name for a model label."""
    base = _slugify(label)
    collisions = [other for other in all_labels if _slugify(other) == base]
    if len(collisions) <= 1:
        return base
    return f"{base}-{index}"


__all__ = ["ReportWorkflowService"]
