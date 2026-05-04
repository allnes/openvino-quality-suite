from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]


def test_docs_portal_layout_matches_expected_docs_architecture():
    required = [
        "mkdocs.yml",
        ".readthedocs.yaml",
        "docs/index.md",
        "docs/start/index.md",
        "docs/start/quickstart.md",
        "docs/start/installation.md",
        "docs/tutorials/first-local-run.md",
        "docs/tutorials/build-a-report-bundle.md",
        "docs/tutorials/compare-against-a-baseline.md",
        "docs/tutorials/run-the-gpu-suite.md",
        "docs/how-to/run-a-suite.md",
        "docs/how-to/validate-a-report.md",
        "docs/how-to/generate-a-metrics-table.md",
        "docs/how-to/use-trend-history.md",
        "docs/how-to/configure-gates.md",
        "docs/how-to/integrate-in-ci.md",
        "docs/how-to/troubleshoot-reporting.md",
        "docs/explanation/architecture.md",
        "docs/explanation/evaluation-surfaces.md",
        "docs/explanation/reporting-model.md",
        "docs/explanation/metric-philosophy.md",
        "docs/reference/cli/index.md",
        "docs/reference/reporting/reporting-spec.md",
        "docs/reference/reporting/bundle-layout.md",
        "docs/reference/reporting/analysis-model.md",
        "docs/reference/schemas/evaluation-report.md",
        "docs/reference/schemas/sample-metric.md",
        "docs/reference/metrics/catalogue.md",
        "docs/reference/metrics/references-and-oracles.md",
        "docs/reference/config/profiles.md",
        "docs/reference/config/gates.md",
        "docs/developers/index.md",
        "docs/developers/adding-a-metric.md",
        "docs/developers/adding-an-analysis-rule.md",
        "docs/developers/adding-a-renderer.md",
        "docs/developers/plugin-entry-points.md",
        "docs/developers/testing-and-quality.md",
        "docs/project/contributing.md",
        "docs/project/code-of-conduct.md",
        "docs/project/changelog.md",
        "docs/project/release-notes.md",
        "docs/project/security.md",
        "docs/assets/mockups/dashboard-mockup.svg",
        "docs/examples/bundles/minimal-likelihood/report.json",
        "docs/examples/bundles/reference-comparison/report.json",
        "docs/examples/bundles/gpu-suite-smoke/report.json",
        "docs/examples/bundles/gated-regression/report.json",
        "docs/examples/bundles/unknown-evidence/report.json",
    ]

    missing = [path for path in required if not (ROOT / path).exists()]

    assert missing == []


def test_legacy_repository_centric_docs_are_removed():
    legacy_paths = [
        "docs/architecture.md",
        "docs/ci_cd.md",
        "docs/data_formats.md",
        "docs/genai_model_matrix.md",
        "docs/gpu_requirements.md",
        "docs/integrations.md",
        "docs/long_context.md",
        "docs/metric_details",
        "docs/metric_playbook.md",
        "docs/metrics.md",
        "docs/openvino_runners.md",
        "docs/rag_agent.md",
        "docs/remote_gpu_from_scratch.md",
        "docs/reports_and_gates.md",
        "docs/skills.md",
        "docs/usage.md",
    ]

    remaining = [path for path in legacy_paths if (ROOT / path).exists()]

    assert remaining == []


def test_mkdocs_uses_strict_nav_validation_with_intentional_generated_exclusions():
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    assert "omitted_files: warn" in config
    assert "absolute_links: relative_to_docs" in config
    assert "not_in_nav: |" in config
    assert "/reference/api/oviqs/**" in config
    assert "openvino-quality-suite.readthedocs.io" in config


def test_legacy_top_level_report_cli_docs_are_removed():
    cli_index = (ROOT / "docs/reference/cli/index.md").read_text(encoding="utf-8")
    legacy_pages = [
        "docs/reference/cli/render-report.md",
        "docs/reference/cli/reference-comparison.md",
    ]

    assert "`oviq compare`" in cli_index
    assert "`oviq render-report`" not in cli_index
    assert "`oviq reference-comparison`" not in cli_index
    assert "`oviq report reference-comparison`" in cli_index
    assert [path for path in legacy_pages if (ROOT / path).exists()] == []


def test_docs_generators_and_workflows_exist():
    required = [
        "scripts/docs/build_cli_reference.py",
        "scripts/docs/build_schema_docs.py",
        "scripts/docs/build_api_pages.py",
        "scripts/docs/build_example_bundles.py",
        ".github/workflows/docs.yml",
        ".github/workflows/docs-preview.yml",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/ISSUE_TEMPLATE/config.yml",
        "CODE_OF_CONDUCT.md",
    ]

    missing = [path for path in required if not (ROOT / path).exists()]

    assert missing == []


def test_curated_docs_pages_are_not_placeholders():
    curated_pages = [
        "docs/index.md",
        "docs/start/index.md",
        "docs/start/quickstart.md",
        "docs/tutorials/first-local-run.md",
        "docs/tutorials/build-a-report-bundle.md",
        "docs/tutorials/compare-against-a-baseline.md",
        "docs/tutorials/run-the-gpu-suite.md",
        "docs/how-to/integrate-in-ci.md",
        "docs/how-to/troubleshoot-reporting.md",
        "docs/explanation/architecture.md",
        "docs/explanation/reporting-model.md",
        "docs/reference/reporting/reporting-spec.md",
        "docs/reference/reporting/bundle-layout.md",
        "docs/reference/reporting/analysis-model.md",
        "docs/developers/adding-a-metric.md",
        "docs/developers/adding-an-analysis-rule.md",
        "docs/developers/adding-a-renderer.md",
        "docs/developers/testing-and-quality.md",
        "docs/project/contributing.md",
    ]

    too_short = [
        path
        for path in curated_pages
        if len((ROOT / path).read_text(encoding="utf-8").splitlines()) < 20
    ]

    assert too_short == []


def test_generated_docs_reference_public_contracts():
    cli_index = (ROOT / "docs/reference/cli/index.md").read_text(encoding="utf-8")
    schema_page = (ROOT / "docs/reference/schemas/evaluation-report.md").read_text(encoding="utf-8")
    sample_schema_page = ROOT / "docs/reference/schemas/sample-metric.md"
    reporting_spec = (ROOT / "docs/reference/reporting/reporting-spec.md").read_text(
        encoding="utf-8"
    )

    assert "`oviq report build`" in cli_index
    assert "schema_version" in schema_page
    assert sample_schema_page.exists()
    assert "unknown" in reporting_spec


def test_example_bundle_reports_validate_against_schema():
    schema = json.loads(
        (ROOT / "src/oviqs/contracts/jsonschema/evaluation_report.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = Draft202012Validator(schema)
    for report_path in (ROOT / "docs/examples/bundles").glob("*/report.json"):
        validator.validate(json.loads(report_path.read_text(encoding="utf-8")))
