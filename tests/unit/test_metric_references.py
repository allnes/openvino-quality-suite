import json

from oviqs.adapters.reporting import CanonicalReportWriter
from oviqs.domain.references import (
    build_report_reference_manifest,
    get_metric_reference,
    list_metric_references,
    require_metric_reference,
)
from oviqs.domain.reports import EvaluationReport, ReportRun


def test_key_metrics_have_degradation_references():
    key_metrics = [
        "nll",
        "perplexity",
        "mean_kl",
        "p95_js",
        "top10_overlap",
        "lost_in_middle_score",
        "distractor_sensitivity",
        "json_valid",
        "prefix_divergence_rate",
        "context_precision",
        "faithfulness",
        "tool_call_validity",
        "redundant_tool_call_rate",
        "task_completion",
    ]
    for metric in key_metrics:
        reference = require_metric_reference(metric)
        assert reference.primary_reference
        assert reference.degradation_rule
        assert reference.sources


def test_reference_manifest_handles_nested_metric_sections():
    report = {
        "serving": {
            "batch_invariance": {
                "mean_kl": 0.0,
                "top10_overlap": 1.0,
            },
            "generation_prefix_divergence": {
                "prefix_divergence_rate": 0.0,
            },
            "status": "pass",
        },
        "generation": {
            "json_validity": {
                "json_valid": True,
                "error": None,
            },
            "ngram_repetition": {
                "n": 3,
                "repetition_rate": 0.0,
                "unique_ngram_ratio": 1.0,
            },
        },
    }
    manifest = build_report_reference_manifest(report)
    assert "batch_invariance" in manifest["serving"]
    assert "generation_prefix_divergence" in manifest["serving"]
    assert "json_validity" in manifest["generation"]
    assert "ngram_repetition" in manifest["generation"]
    assert "_warnings" not in manifest


def test_write_report_populates_metric_references(tmp_path):
    out = tmp_path / "report.json"
    report = EvaluationReport(
        run=ReportRun(id="r1"),
        likelihood={"nll": 1.0, "perplexity": 2.718, "num_tokens": 8},
        rag={"context_precision": 1.0, "supported_claim_ratio": None},
    )
    CanonicalReportWriter().write(report, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "metric_references" in payload
    assert "nll" in payload["metric_references"]["likelihood"]
    assert "context_precision" in payload["metric_references"]["rag"]
    assert "supported_claim_ratio" in payload["metric_references"]["rag"]


def test_reference_catalog_has_unique_metric_names():
    seen = {}
    for reference in list_metric_references():
        for metric_name in reference.metric_names:
            assert metric_name not in seen, f"{metric_name} duplicated in reference registry"
            seen[metric_name] = reference.family
            assert get_metric_reference(metric_name) is reference
