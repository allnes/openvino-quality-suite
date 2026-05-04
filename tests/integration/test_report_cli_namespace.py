from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from oviqs.cli import app


@pytest.fixture()
def report_path(tmp_path: Path) -> Path:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "openvino_llm_quality_v1",
                "run": {
                    "id": "run1",
                    "suite": "openvino_llm_quality_v1",
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
                "summary": {"overall_status": "unknown", "main_findings": []},
                "likelihood": {"status": "unknown", "perplexity": 2.0},
            }
        ),
        encoding="utf-8",
    )
    return report


@pytest.fixture()
def report_bundle(tmp_path: Path, report_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    result = CliRunner().invoke(
        app,
        ["report", "build", "--report", str(report_path), "--out", str(bundle)],
    )
    assert result.exit_code == 0, result.output
    return bundle


def test_report_namespace_build_writes_bundle(report_bundle: Path):
    assert (report_bundle / "report.json").exists()
    assert (report_bundle / "analysis.json").exists()
    assert (report_bundle / "metrics.csv").exists()
    assert (report_bundle / "dashboard.html").exists()


def test_report_namespace_validates_bundle_report(report_bundle: Path):
    result = CliRunner().invoke(
        app,
        ["report", "validate", "--report", str(report_bundle / "report.json")],
    )
    assert result.exit_code == 0, result.output


def test_report_namespace_renders_markdown(tmp_path: Path, report_bundle: Path):
    rendered = tmp_path / "rendered.md"
    result = CliRunner().invoke(
        app,
        [
            "report",
            "render",
            "--bundle",
            str(report_bundle),
            "--format",
            "markdown",
            "--out",
            str(rendered),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "OVIQS Report" in rendered.read_text(encoding="utf-8")


def test_report_namespace_writes_metrics_table(tmp_path: Path, report_path: Path):
    metrics = tmp_path / "metrics.csv"
    result = CliRunner().invoke(
        app,
        ["report", "metrics-table", "--report", str(report_path), "--out", str(metrics)],
    )
    assert result.exit_code == 0, result.output
    assert "likelihood.perplexity" in metrics.read_text(encoding="utf-8")


def test_top_level_compare_remains_available():
    result = CliRunner().invoke(app, ["compare", "--help"])

    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("command", ["render-report", "reference-comparison"])
def test_legacy_top_level_report_commands_are_removed(command: str):
    result = CliRunner().invoke(app, [command, "--help"])

    assert result.exit_code != 0
