from __future__ import annotations

from oviqs.application.reporting import (
    build_report_interface_response,
    validate_evaluation_report_contract,
    validate_metric_observation_contract,
    validate_report_bundle_metadata,
    validate_sample_metric_contract,
)


def _contract_report() -> dict:
    return {
        "schema_version": "openvino_llm_quality_v1",
        "run": {
            "id": "contract-run",
            "suite": "contract-suite",
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
            "sections": {
                "likelihood": {
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
    }


def test_evaluation_report_contract_accepts_enriched_reporting_payload():
    response = build_report_interface_response(_contract_report())

    errors = validate_evaluation_report_contract(response.report)

    assert errors == []
    assert response.report["analysis"]["metrics"][0]["path"] == "likelihood.perplexity"


def test_interface_response_metrics_are_metric_observation_contract_rows():
    response = build_report_interface_response(_contract_report(), report_uri="reports/run.json")
    http_payload = response.http_payload()
    grpc_payload = response.grpc_mapping()

    assert http_payload["metrics"] == grpc_payload["metrics"]
    assert grpc_payload["report_uri"] == "reports/run.json"
    assert validate_metric_observation_contract(http_payload["metrics"][0]) == []


def test_report_bundle_contract_requires_stable_artifact_layout():
    metadata: dict[str, object] = {
        "root": "report-bundle",
        "report_json": "report.json",
        "analysis_json": "analysis.json",
        "metrics_csv": "metrics.csv",
        "sample_metrics_jsonl": "sample_metrics.jsonl",
        "index_md": "index.md",
        "dashboard_html": "dashboard.html",
        "metadata_json": "metadata.json",
    }

    assert validate_report_bundle_metadata(metadata) == []
    metadata["metadata_json"] = None
    assert validate_report_bundle_metadata(metadata) == []


def test_sample_metric_contract_keeps_sample_level_rows_machine_readable():
    row = {
        "section": "likelihood",
        "sample_index": 0,
        "sample_id": "sample-0",
        "status": "warning",
        "metric": "perplexity",
        "value": 10.0,
    }

    assert validate_sample_metric_contract(row) == []


def test_non_current_report_payloads_are_rejected():
    old_payload = {
        "id": "old-run",
        "suite": "old-suite",
        "created_at": "2026-05-03T00:00:00Z",
        "model_id": "old-model",
        "overall_status": "pass",
        "likelihood": {"status": "pass", "perplexity": 4.0},
    }

    assert validate_evaluation_report_contract(old_payload) == [
        "Unsupported EvaluationReport schema_version: None"
    ]
