"""Tests for the multi-model suite bundle (oviq report build-suite)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from oviqs.cli import app
from oviqs.domain.reporting import build_suite_model_summary
from oviqs.platform.bootstrap import build_default_container


def _report(model: str, *, ppl: float, mean_kl: float) -> dict:
    return {
        "schema_version": "openvino_llm_quality_v1",
        "run": {"id": f"{model}-run", "model": model, "reference_model": f"{model}-pt"},
        "summary": {"overall_status": "warning"},
        "likelihood": {"status": "pass", "perplexity": ppl, "nll": 1.0},
        "inference_equivalence": {"status": "pass", "mean_kl": mean_kl, "mean_logit_cosine": 0.99},
        "performance": {"status": "pass", "forward_latency_ms_mean": 30.0},
    }


def test_build_suite_model_summary_extracts_headline_figures():
    report = _report("m1", ppl=9.8, mean_kl=0.018)
    summary = build_suite_model_summary("Mistral", report, "models/mistral")
    assert summary.label == "Mistral"
    assert summary.bundle_dir == "models/mistral"
    assert summary.perplexity == 9.8
    assert summary.mean_kl == 0.018
    assert summary.overall_status == "warning"


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_suite_bundle_writes_index_and_per_model_bundles(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write(a, _report("alpha", ppl=29.7, mean_kl=0.28))
    _write(b, _report("beta", ppl=9.8, mean_kl=0.018))
    out = tmp_path / "suite"

    workflow = build_default_container().report_workflow_service()
    suite = workflow.build_suite_bundle([f"Alpha={a}", f"Beta={b}"], out)

    index = Path(suite.index_html)
    assert index.exists()
    html = index.read_text(encoding="utf-8")
    assert "Alpha" in html and "Beta" in html
    assert "comparison.html" in html
    # per-model bundles built with full dashboards
    assert (out / "models" / "alpha" / "dashboard.html").exists()
    assert (out / "models" / "beta" / "dashboard.html").exists()
    assert (out / "comparison.html").exists()
    assert len(suite.model_bundles) == 2


def test_report_build_suite_cli(tmp_path):
    a = tmp_path / "a.json"
    _write(a, _report("alpha", ppl=29.7, mean_kl=0.28))
    out = tmp_path / "suite"
    result = CliRunner().invoke(
        app, ["report", "build-suite", "--report", f"Alpha={a}", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert (out / "index.html").exists()
