import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from oviqs.adapters.analysis import BuiltInAnalysisRules, LocalTrendStore, MetricReferenceCatalog
from oviqs.adapters.reporting import (
    CsvMetricsWriter,
    HtmlDashboardRenderer,
    JsonReportAdapter,
    MarkdownReportRenderer,
)
from oviqs.application.reporting import (
    ReportAnalysisService,
    ReportComparisonService,
    build_report_interface_response,
    validate_evaluation_report_contract,
    validate_sample_metric_contract,
)
from oviqs.application.reporting.normalization import flatten_report_metrics
from oviqs.cli import app
from oviqs.domain.reporting import MetricPath, worst_status
from oviqs.domain.reporting.view_model import build_report_view_model


def _analysis_service() -> ReportAnalysisService:
    return ReportAnalysisService(
        rules=[BuiltInAnalysisRules()],
        metric_catalog=MetricReferenceCatalog(),
    )


def _snapshot_report() -> dict:
    return {
        "schema_version": "openvino_llm_quality_v1",
        "run": {
            "id": "snapshot-run",
            "suite": "snapshot-suite",
            "model": "m",
            "reference": "cpu",
            "current": "gpu",
            "device": "GPU",
            "precision": "FP16",
            "created_at": "2026-05-03T00:00:00Z",
        },
        "summary": {"overall_status": "warning", "main_findings": ["needs review"]},
        "likelihood": {"status": "warning", "perplexity": 10.0},
        "gates": {
            "overall_status": "warning",
            "sections": {
                "likelihood": {
                    "status": "warning",
                    "checks": {
                        "perplexity_max": {
                            "metric": "perplexity",
                            "metric_path": "perplexity",
                            "value": 10.0,
                            "threshold": 8.0,
                            "reference_status": "present",
                            "status": "warning",
                        }
                    },
                }
            },
        },
        "reproducibility": {"seed": 7, "versions": {"oviqs": "test"}},
    }


def test_metric_path_parsing_and_status_ordering():
    path = MetricPath.parse("serving.batch_invariance.mean_kl")

    assert path.section == "serving"
    assert path.name == "mean_kl"
    assert path.dotted == "serving.batch_invariance.mean_kl"
    assert worst_status(["pass", "unknown", "warning"]) == "warning"


def test_flatten_report_metrics_keeps_unknown_gated_metrics_unknown():
    report = {
        "run": {"id": "r1"},
        "summary": {"overall_status": "warning"},
        "likelihood": {"status": "pass", "perplexity": 10.0},
        "gates": {
            "sections": {
                "likelihood": {
                    "checks": {
                        "perplexity": {"status": "pass", "threshold": 12.0, "rule": "lte"},
                        "nll": {"status": "unknown", "threshold": 3.0, "rule": "lte"},
                    }
                }
            }
        },
    }

    metrics = {metric.path: metric for metric in flatten_report_metrics(report)}

    assert metrics["likelihood.perplexity"].status == "pass"
    assert metrics["likelihood.perplexity"].threshold == 12.0
    assert metrics["likelihood.nll"].status == "unknown"
    assert metrics["likelihood.nll"].value is None


def test_report_interface_response_uses_canonical_analysis_contract():
    response = build_report_interface_response(
        _snapshot_report(),
        analysis_service=_analysis_service(),
    )
    http_payload = response.http_payload()
    grpc_payload = response.grpc_mapping()

    assert response.run_id == "snapshot-run"
    assert http_payload["report"]["analysis"]["summary"]["overall_status"] == "warning"
    assert http_payload["metrics"][0]["path"]
    assert grpc_payload["metrics"] == http_payload["metrics"]
    assert "reference_id" in grpc_payload["metrics"][0]


def test_analysis_and_comparison_include_baseline_deltas():
    baseline = {
        "run": {"id": "base"},
        "summary": {"overall_status": "pass"},
        "likelihood": {"status": "pass", "perplexity": 8.0},
    }
    current = {
        "run": {"id": "cur"},
        "summary": {"overall_status": "warning"},
        "likelihood": {"status": "warning", "perplexity": 10.0},
    }

    comparison = ReportComparisonService().compare(current, baseline)
    analysis = _analysis_service().analyze(current, baseline=baseline)

    assert comparison.biggest_regressions[0].path == "likelihood.perplexity"
    assert comparison.biggest_regressions[0].delta_abs == 2.0
    assert analysis.summary.warning >= 1
    assert analysis.findings


def test_comparison_can_use_local_trend_store_as_baseline(tmp_path: Path):
    baseline = {
        "run": {"id": "base"},
        "summary": {"overall_status": "pass"},
        "likelihood": {"status": "pass", "perplexity": 8.0},
    }
    current = {
        "run": {"id": "cur"},
        "summary": {"overall_status": "warning"},
        "likelihood": {"status": "warning", "perplexity": 10.0},
    }
    trend_store = LocalTrendStore(history_path=tmp_path / "report-history.jsonl")
    trend_store.append(baseline)

    comparison = ReportComparisonService(trend_store).compare(current)
    analysis = ReportAnalysisService(
        rules=[BuiltInAnalysisRules()],
        metric_catalog=MetricReferenceCatalog(),
        trend_store=trend_store,
    ).analyze(current)

    assert comparison.biggest_regressions[0].path == "likelihood.perplexity"
    assert comparison.biggest_regressions[0].baseline_value == 8.0
    assert comparison.trend_points == (
        {
            "path": "likelihood.perplexity",
            "run_id": "base",
            "status": "pass",
            "value": 8.0,
        },
    )
    assert analysis.trend_points == comparison.trend_points
    assert analysis.to_dict()["trend_points"] == list(comparison.trend_points)


def test_report_cli_can_use_trend_history_as_baseline(tmp_path: Path):
    history = tmp_path / "history.jsonl"
    current = tmp_path / "current.json"
    analysis = tmp_path / "analysis.json"
    history.write_text(
        json.dumps(
            {
                "schema_version": "openvino_llm_quality_v1",
                "run": {
                    "id": "baseline-run",
                    "suite": "suite",
                    "created_at": "2026-05-03T00:00:00Z",
                },
                "summary": {"overall_status": "pass"},
                "likelihood": {"status": "pass", "perplexity": 8.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    current.write_text(
        json.dumps(
            {
                "schema_version": "openvino_llm_quality_v1",
                "run": {
                    "id": "current-run",
                    "suite": "suite",
                    "created_at": "2026-05-03T01:00:00Z",
                },
                "summary": {"overall_status": "warning"},
                "likelihood": {"status": "warning", "perplexity": 10.0},
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "report",
            "analyze",
            "--report",
            str(current),
            "--trend-history",
            str(history),
            "--out",
            str(analysis),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(analysis.read_text(encoding="utf-8"))
    assert payload["biggest_regressions"][0]["path"] == "likelihood.perplexity"
    assert payload["biggest_regressions"][0]["baseline_value"] == 8.0
    assert payload["trend_points"] == [
        {
            "path": "likelihood.perplexity",
            "run_id": "baseline-run",
            "status": "pass",
            "value": 8.0,
        }
    ]


def test_analysis_rules_emit_category_findings_for_gated_metrics():
    current = {
        "run": {"id": "cur"},
        "summary": {"overall_status": "warning"},
        "serving": {"status": "pass", "batch_mean_kl": 0.2},
        "rag": {"status": "pass", "citation_recall": 0.4},
        "agent": {"status": "pass", "recovery_score": 0.2},
        "gates": {
            "sections": {
                "serving": {
                    "checks": {
                        "batch_mean_kl_max": {
                            "metric": "batch_mean_kl",
                            "metric_path": "batch_mean_kl",
                            "status": "warning",
                            "threshold": 0.01,
                        }
                    }
                },
                "rag": {
                    "checks": {
                        "citation_recall_min": {
                            "metric": "citation_recall",
                            "metric_path": "citation_recall",
                            "status": "warning",
                            "threshold": 0.9,
                        }
                    }
                },
                "agent": {
                    "checks": {
                        "recovery_score_min": {
                            "metric": "recovery_score",
                            "metric_path": "recovery_score",
                            "status": "warning",
                            "threshold": 0.8,
                        }
                    }
                },
            }
        },
    }

    analysis = _analysis_service().analyze(current)
    finding_ids = {finding.id for finding in analysis.findings}

    assert "serving-batch-invariance" in finding_ids
    assert "rag-citation-grounding" in finding_ids
    assert "agent-recovery-tool-use" in finding_ids


def test_analysis_extracts_sample_outliers():
    report = {
        "run": {"id": "cur"},
        "summary": {"overall_status": "warning"},
        "likelihood": {
            "status": "warning",
            "samples": [
                {"id": "s1", "perplexity": 10.0},
                {"id": "s2", "perplexity": 11.0},
                {"id": "s3", "perplexity": 100.0},
            ],
        },
    }

    analysis = _analysis_service().analyze(report)

    assert analysis.sample_outliers
    assert analysis.sample_outliers[0]["sample_id"] == "s3"
    assert analysis.sample_outliers[0]["metric"] == "perplexity"


def test_reporting_renderers_match_golden_snapshots(tmp_path: Path):
    report = _snapshot_report()
    analysis = _analysis_service().analyze(report)
    report["analysis"] = analysis.to_dict()
    view_model = build_report_view_model(report, analysis)
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "reports"

    markdown = MarkdownReportRenderer().render(view_model)
    assert markdown == (fixtures / "golden_reporting_markdown.md").read_text(encoding="utf-8")

    html = HtmlDashboardRenderer().render(view_model)
    for expected in (
        (fixtures / "golden_reporting_html_smoke.txt").read_text(encoding="utf-8").splitlines()
    ):
        assert expected in html

    metrics_csv = tmp_path / "metrics.csv"
    CsvMetricsWriter().write(list(_analysis_service().analyze(report).metrics), metrics_csv)
    assert metrics_csv.read_text(encoding="utf-8") == (
        fixtures / "golden_reporting_metrics.csv"
    ).read_text(encoding="utf-8")


def _write_bundle_cli_fixture(tmp_path: Path) -> tuple[Path, Path]:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "openvino_llm_quality_v1",
                "run": {"id": "run1", "suite": "suite", "created_at": "2026-05-03T00:00:00Z"},
                "summary": {"overall_status": "warning", "main_findings": ["needs review"]},
                "likelihood": {
                    "status": "warning",
                    "perplexity": 10.0,
                    "samples": [
                        {"id": "s1", "perplexity": 10.0},
                        {"id": "s2", "perplexity": 11.0},
                        {"id": "s3", "perplexity": 100.0},
                    ],
                },
                "metric_coverage": {
                    "entries": [
                        {"section": "likelihood", "metric": "perplexity", "status": "measured"},
                        {"section": "serving", "metric": "kv_mean_kl", "status": "unknown"},
                    ]
                },
                "reproducibility": {
                    "seed": 7,
                    "versions": {"openvino": "test", "oviqs": "test"},
                },
            }
        ),
        encoding="utf-8",
    )
    gates = tmp_path / "gates.yaml"
    gates.write_text("likelihood:\n  perplexity_max: 8.0\n", encoding="utf-8")
    return report, gates


def _assert_bundle_outputs(bundle: Path) -> None:
    assert (bundle / "report.json").exists()
    assert (bundle / "analysis.json").exists()
    assert (bundle / "metrics.csv").read_text(encoding="utf-8").startswith("path,section")
    assert (bundle / "sample_metrics.jsonl").read_text(encoding="utf-8").strip()
    assert (bundle / "assets").is_dir()
    bundled_report = json.loads((bundle / "report.json").read_text(encoding="utf-8"))
    bundle_metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    assert bundled_report["gates"]["overall_status"] == "warning"
    assert bundled_report["analysis"]["sample_outliers"][0]["sample_id"] == "s3"
    assert bundle_metadata["sample_metrics_validation_errors"] == []
    markdown = (bundle / "index.md").read_text(encoding="utf-8")
    dashboard = (bundle / "dashboard.html").read_text(encoding="utf-8")
    assert "Executive Summary" in markdown
    assert "Gate Results" in markdown
    assert "Reproducibility" in markdown
    assert "Sample-Level Outliers" in markdown
    assert "<!doctype html>" in dashboard
    assert "Sample-Level Outliers" in dashboard
    assert "statusChart" in dashboard


def _invoke_report_render(runner: CliRunner, bundle: Path, out: Path, format_name: str):
    return runner.invoke(
        app,
        [
            "report",
            "render",
            "--bundle",
            str(bundle),
            "--format",
            format_name,
            "--out",
            str(out),
        ],
    )


def _assert_report_analysis_cli(
    runner: CliRunner, report: Path, gates: Path, tmp_path: Path
) -> None:
    analysis_json = tmp_path / "analysis.json"
    result = runner.invoke(
        app,
        [
            "report",
            "analyze",
            "--report",
            str(report),
            "--gates",
            str(gates),
            "--out",
            str(analysis_json),
        ],
    )
    assert result.exit_code == 0, result.output
    analysis_payload = json.loads(analysis_json.read_text(encoding="utf-8"))
    assert analysis_payload["metrics"][0]["threshold"] == 8.0


def _assert_report_metrics_and_validation_cli(
    runner: CliRunner,
    report: Path,
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.csv"
    result = runner.invoke(
        app,
        ["report", "metrics-table", "--report", str(report), "--out", str(metrics)],
    )
    assert result.exit_code == 0, result.output
    assert "likelihood.perplexity" in metrics.read_text(encoding="utf-8")

    result = runner.invoke(app, ["report", "validate", "--report", str(report)])
    assert result.exit_code == 0, result.output
    assert "structurally valid" in result.output


def _assert_unsupported_report_validation_fails(runner: CliRunner, tmp_path: Path) -> None:
    unsupported = tmp_path / "unsupported.json"
    unsupported.write_text(
        json.dumps({"schema_version": "future", "run": {"id": "bad"}, "summary": {}}),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["report", "validate", "--report", str(unsupported)])
    assert result.exit_code != 0
    assert "Unsupported EvaluationReport schema_version" in result.output


def test_report_cli_builds_bundle_and_renders_outputs_with_yaml_gates(tmp_path: Path):
    report, gates = _write_bundle_cli_fixture(tmp_path)
    bundle = tmp_path / "bundle"
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["report", "build", "--report", str(report), "--gates", str(gates), "--out", str(bundle)],
    )

    assert result.exit_code == 0, result.output
    _assert_bundle_outputs(bundle)

    rendered = tmp_path / "rendered.md"
    result = _invoke_report_render(runner, bundle, rendered, "markdown")
    assert result.exit_code == 0, result.output
    assert "Metric Details" in rendered.read_text(encoding="utf-8")

    rendered_html = tmp_path / "rendered.html"
    result = _invoke_report_render(runner, bundle, rendered_html, "html-dashboard")
    assert result.exit_code == 0, result.output
    assert "Biggest Regressions" in rendered_html.read_text(encoding="utf-8")

    _assert_report_analysis_cli(runner, report, gates, tmp_path)
    _assert_report_metrics_and_validation_cli(runner, report, tmp_path)
    _assert_unsupported_report_validation_fails(runner, tmp_path)


def test_sample_metric_contract_validation():
    assert (
        validate_sample_metric_contract(
            {"section": "likelihood", "sample_index": 0, "perplexity": 2.0}
        )
        == []
    )

    errors = validate_sample_metric_contract({"section": "", "sample_index": "0", "status": "bad"})

    assert any("section" in error for error in errors)
    assert any("sample_index" in error for error in errors)
    assert any("status" in error for error in errors)


def test_evaluation_report_schema_validates_known_analysis_fields():
    report = {
        "schema_version": "openvino_llm_quality_v1",
        "run": {"id": "run1", "suite": "suite", "created_at": "2026-05-03T00:00:00Z"},
        "summary": {"overall_status": "pass"},
        "analysis": {
            "metrics": [
                {
                    "path": "likelihood.perplexity",
                    "section": "likelihood",
                    "name": "perplexity",
                    "status": "pass",
                    "severity": "none",
                }
            ],
            "findings": [
                {
                    "id": "f1",
                    "title": "bad severity",
                    "severity": "urgent",
                    "category": "likelihood",
                    "status": "warning",
                }
            ],
        },
    }

    errors = validate_evaluation_report_contract(report)

    assert any("urgent" in error or "severity" in error for error in errors)


def test_report_schema_normalization_rejects_non_current_payloads(tmp_path: Path):
    old_payload = {
        "id": "old-run",
        "status": "pass",
        "model_id": "old-model",
        "likelihood": {"status": "pass", "ppl": 2.0},
    }

    errors = validate_evaluation_report_contract(old_payload)

    assert "Unsupported EvaluationReport schema_version: None" in errors

    report_path = tmp_path / "old.json"
    report_path.write_text(json.dumps(old_payload), encoding="utf-8")
    try:
        JsonReportAdapter().load(report_path)
    except ValueError as exc:
        assert "Unsupported EvaluationReport schema_version" in str(exc)
    else:
        raise AssertionError("expected non-current report payload to be rejected")


def test_reporting_imports_keep_heavy_optional_dependencies_lazy():
    script = (
        "import sys\n"
        "import oviqs.application.reporting\n"
        "import oviqs.adapters.reporting\n"
        "heavy = {'openvino', 'torch', 'transformers', 'optimum', 'pandas', 'jsonschema'}\n"
        "print('\\n'.join(sorted(heavy.intersection(sys.modules))))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "\n"
