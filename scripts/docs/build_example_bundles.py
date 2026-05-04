from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "examples" / "bundles"
SCHEMA_DIR = ROOT / "src" / "oviqs" / "contracts" / "jsonschema"


def main() -> None:
    report_validator = Draft202012Validator(
        json.loads((SCHEMA_DIR / "evaluation_report.schema.json").read_text(encoding="utf-8"))
    )
    bundle_validator = Draft202012Validator(
        json.loads((SCHEMA_DIR / "report_bundle.schema.json").read_text(encoding="utf-8"))
    )
    for bundle in example_bundles():
        write_bundle(bundle, report_validator, bundle_validator)


def example_bundles() -> list[dict[str, Any]]:
    return [
        {
            "slug": "minimal-likelihood",
            "title": "Minimal likelihood report",
            "description": "Smallest useful likelihood report with unknown evidence status.",
            "command": [
                "oviq eval-likelihood \\",
                "  --model dummy \\",
                "  --backend dummy \\",
                "  --dataset data/likelihood.jsonl \\",
                "  --out report.json",
            ],
            "review_notes": [
                "`summary.overall_status` is `unknown` because this fixture demonstrates "
                "missing evidence.",
                "`metrics.csv` contains one scalar path: `likelihood.perplexity`.",
                "Use this bundle as the smallest schema-validation and renderer smoke fixture.",
            ],
            "report": base_report(
                "docs-minimal-likelihood",
                "unknown",
                ["Minimal deterministic documentation fixture."],
                "CPU",
                {
                    "status": "unknown",
                    "nll": 1.0,
                    "perplexity": 2.718281828,
                    "num_tokens": 4,
                },
            ),
            "metrics": [
                ("likelihood.perplexity", "likelihood", "perplexity", 2.718281828, "unknown"),
            ],
            "samples": [
                {
                    "section": "likelihood",
                    "sample_index": 0,
                    "sample_id": "prompt-0001",
                    "metric": "perplexity",
                    "value": 2.718281828,
                    "status": "unknown",
                }
            ],
            "findings": [],
            "baseline": None,
        },
        {
            "slug": "reference-comparison",
            "title": "Reference comparison report",
            "description": "Two-run comparison fixture with a baseline report reference.",
            "command": [
                "oviq report reference-comparison \\",
                "  --report baseline=baseline.json \\",
                "  --report current=report.json \\",
                "  --out comparison.md",
            ],
            "review_notes": [
                "`run.reference` and `run.current` identify the compared runs.",
                "`analysis.json` includes a passing finding for `likelihood.perplexity`.",
                "Use this bundle when checking comparison tables and baseline labels.",
            ],
            "report": base_report(
                "docs-reference-current",
                "pass",
                ["Current run stays inside the documented reference threshold."],
                "CPU",
                {
                    "status": "pass",
                    "nll": 0.91,
                    "perplexity": 2.484322533,
                    "num_tokens": 1024,
                },
                reference="docs-reference-baseline",
            ),
            "metrics": [
                ("likelihood.perplexity", "likelihood", "perplexity", 2.484322533, "pass"),
                ("likelihood.nll", "likelihood", "nll", 0.91, "pass"),
            ],
            "samples": [
                {
                    "section": "likelihood",
                    "sample_index": 0,
                    "sample_id": "prompt-0001",
                    "metric": "negative_log_likelihood",
                    "value": 0.88,
                    "status": "pass",
                },
                {
                    "section": "likelihood",
                    "sample_index": 1,
                    "sample_id": "prompt-0002",
                    "metric": "negative_log_likelihood",
                    "value": 0.94,
                    "status": "pass",
                },
            ],
            "findings": [
                {
                    "path": "likelihood.perplexity",
                    "status": "pass",
                    "message": "Perplexity remains within the reference threshold.",
                }
            ],
            "baseline": "baseline.json",
        },
        {
            "slug": "gpu-suite-smoke",
            "title": "GPU suite smoke report",
            "description": "GPU fixture with Runtime and GenAI evidence recorded in one report.",
            "command": [
                "oviq run-gpu-suite \\",
                "  --model models/tiny-ov \\",
                "  --genai-model models/tiny-genai-ov \\",
                "  --dataset data/likelihood.jsonl \\",
                "  --out report.json \\",
                "  --device GPU",
            ],
            "review_notes": [
                "`run.device` is `GPU` so device-sensitive reporting paths can be tested.",
                "`likelihood.generation.status` remains `unknown` when GenAI logits are "
                "unavailable.",
                "Use this bundle for GPU docs without requiring a real device in "
                "documentation tests.",
            ],
            "report": base_report(
                "docs-gpu-suite-smoke",
                "warning",
                ["GPU runtime checks passed; GenAI aligned logits were unavailable."],
                "GPU",
                {
                    "status": "warning",
                    "nll": 1.08,
                    "perplexity": 2.944679551,
                    "num_tokens": 2048,
                    "generation": {
                        "status": "unknown",
                        "reason": "OpenVINO GenAI generation path does not expose logits.",
                    },
                },
            ),
            "metrics": [
                ("likelihood.perplexity", "likelihood", "perplexity", 2.944679551, "warning"),
                (
                    "likelihood.generation.status",
                    "likelihood",
                    "generation.status",
                    "unknown",
                    "unknown",
                ),
            ],
            "samples": [
                {
                    "section": "likelihood",
                    "sample_index": 0,
                    "sample_id": "gpu-prompt-0001",
                    "metric": "perplexity",
                    "value": 2.944679551,
                    "status": "warning",
                }
            ],
            "findings": [
                {
                    "path": "likelihood.generation.status",
                    "status": "unknown",
                    "message": "Generation-only backend cannot provide aligned logits.",
                }
            ],
            "baseline": None,
        },
        {
            "slug": "gated-regression",
            "title": "Gated regression report",
            "description": "Failing bundle that demonstrates gate output and severity.",
            "command": [
                "oviq report analyze \\",
                "  --report report.json \\",
                "  --baseline baseline.json \\",
                "  --gates gates.yaml \\",
                "  --out analysis.json",
            ],
            "review_notes": [
                "`summary.overall_status` is `fail` and the primary finding has high severity.",
                "`metrics.csv` includes both failing and warning likelihood rows.",
                "Use this bundle when validating dashboards, gate copy and CI failure summaries.",
            ],
            "report": base_report(
                "docs-gated-regression",
                "fail",
                ["Perplexity regressed beyond the configured gate."],
                "CPU",
                {
                    "status": "fail",
                    "nll": 1.37,
                    "perplexity": 3.935480069,
                    "num_tokens": 1024,
                },
            ),
            "metrics": [
                ("likelihood.perplexity", "likelihood", "perplexity", 3.935480069, "fail"),
                ("likelihood.nll", "likelihood", "nll", 1.37, "warning"),
            ],
            "samples": [
                {
                    "section": "likelihood",
                    "sample_index": 0,
                    "sample_id": "regression-0001",
                    "metric": "negative_log_likelihood",
                    "value": 1.37,
                    "status": "fail",
                }
            ],
            "findings": [
                {
                    "path": "likelihood.perplexity",
                    "status": "fail",
                    "severity": "high",
                    "message": "Perplexity exceeded the maximum allowed value.",
                }
            ],
            "baseline": "baseline.json",
        },
        {
            "slug": "unknown-evidence",
            "title": "Unknown evidence report",
            "description": "Fixture showing how unsupported evidence is preserved as unknown.",
            "command": [
                "oviq eval-likelihood \\",
                "  --model models/generation-only \\",
                "  --backend ov-genai \\",
                "  --dataset data/likelihood.jsonl \\",
                "  --out report.json",
            ],
            "review_notes": [
                "The report has no substituted likelihood value because aligned logits are absent.",
                "`analysis.json` explains the unsupported evidence path instead of marking "
                "it pass.",
                "Use this bundle when testing mandatory gates over missing evidence.",
            ],
            "report": base_report(
                "docs-unknown-evidence",
                "unknown",
                ["Backend completed generation but did not expose likelihood evidence."],
                "GPU",
                {
                    "status": "unknown",
                    "reason": "No aligned token log probabilities were available.",
                },
            ),
            "metrics": [
                ("likelihood.status", "likelihood", "status", "unknown", "unknown"),
            ],
            "samples": [
                {
                    "section": "likelihood",
                    "sample_index": 0,
                    "sample_id": "unknown-0001",
                    "metric": "negative_log_likelihood",
                    "value": None,
                    "status": "unknown",
                }
            ],
            "findings": [
                {
                    "path": "likelihood",
                    "status": "unknown",
                    "message": "Evidence was unavailable and was not substituted.",
                }
            ],
            "baseline": None,
        },
    ]


def base_report(
    run_id: str,
    status: str,
    findings: list[str],
    device: str,
    likelihood: dict[str, Any],
    *,
    reference: str | None = None,
) -> dict[str, Any]:
    run: dict[str, Any] = {
        "id": run_id,
        "suite": "openvino_llm_quality_v1",
        "model": "docs-model",
        "device": device,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    if reference is not None:
        run["reference"] = reference
        run["current"] = run_id
    return {
        "schema_version": "openvino_llm_quality_v1",
        "run": run,
        "summary": {
            "overall_status": status,
            "main_findings": findings,
        },
        "likelihood": likelihood,
        "metric_references": {
            "likelihood.perplexity": {
                "oracle": "lower_is_better",
                "warning": 3.0,
                "fail": 3.5,
            }
        },
        "reproducibility": {"fixture": "docs"},
    }


def write_bundle(
    bundle: dict[str, Any],
    report_validator: Draft202012Validator,
    bundle_validator: Draft202012Validator,
) -> None:
    directory = OUT_DIR / str(bundle["slug"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "assets").mkdir(exist_ok=True)

    report = bundle["report"]
    report_validator.validate(report)
    metadata = {
        "root": ".",
        "report_json": "report.json",
        "analysis_json": "analysis.json",
        "metrics_csv": "metrics.csv",
        "sample_metrics_jsonl": "sample_metrics.jsonl",
        "index_md": "index.md",
        "dashboard_html": "dashboard.html",
        "metadata_json": "metadata.json",
        "created_at": "2026-01-01T00:00:00+00:00",
        "schema_version": "openvino_llm_quality_v1",
        "source_report": "report.json",
        "baseline_report": bundle["baseline"],
    }
    bundle_validator.validate(metadata)

    write_json(directory / "report.json", report)
    write_json(directory / "analysis.json", analysis_payload(bundle))
    write_json(directory / "metadata.json", metadata)
    (directory / "metrics.csv").write_text(metrics_csv(bundle["metrics"]), encoding="utf-8")
    (directory / "sample_metrics.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in bundle["samples"]) + "\n",
        encoding="utf-8",
    )
    (directory / "index.md").write_text(index_markdown(bundle), encoding="utf-8")
    (directory / "dashboard.html").write_text(dashboard_html(bundle), encoding="utf-8")


def analysis_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "openvino_report_analysis_v1",
        "findings": bundle["findings"],
        "metrics": [
            {
                "path": path,
                "section": section,
                "name": name,
                "value": value,
                "status": status,
            }
            for path, section, name, value, status in bundle["metrics"]
        ],
    }


def metrics_csv(metrics: list[tuple[str, str, str, Any, str]]) -> str:
    rows = ["path,section,name,value,status"]
    rows.extend(",".join(map(str, metric)) for metric in metrics)
    return "\n".join(rows) + "\n"


def index_markdown(bundle: dict[str, Any]) -> str:
    report = bundle["report"]
    status = report["summary"]["overall_status"]
    metric_rows = [
        f"| `{path}` | `{section}` | `{name}` | `{status}` |"
        for path, section, name, _value, status in bundle["metrics"]
    ]
    review_notes = [f"- {note}" for note in bundle["review_notes"]]
    return "\n".join(
        [
            f"# {bundle['title']}",
            "",
            str(bundle["description"]),
            "",
            "## Reproduce the scenario",
            "",
            "```bash",
            *bundle["command"],
            "```",
            "",
            "The command documents the intended workflow for this fixture. The generated "
            "files in this directory are deterministic examples, not benchmark claims.",
            "",
            "## Bundle contents",
            "",
            f"- Run id: `{report['run']['id']}`",
            f"- Overall status: `{status}`",
            "- Files: `report.json`, `analysis.json`, `metrics.csv`, "
            "`sample_metrics.jsonl`, `metadata.json`, `dashboard.html`",
            "",
            "## Metric rows",
            "",
            "| Path | Section | Name | Status |",
            "|---|---|---|---|",
            *metric_rows,
            "",
            "## Review notes",
            "",
            *review_notes,
            "",
            "Use this fixture when checking documentation examples, schema validation, "
            "renderer behavior and reporting copy without relying on external model downloads.",
            "",
        ]
    )


def dashboard_html(bundle: dict[str, Any]) -> str:
    report = bundle["report"]
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f"<title>{bundle['title']}</title></head><body>"
        f"<h1>{bundle['title']}</h1>"
        f"<p>{bundle['description']}</p>"
        f"<p>Status: <strong>{report['summary']['overall_status']}</strong></p>"
        "</body></html>\n"
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
