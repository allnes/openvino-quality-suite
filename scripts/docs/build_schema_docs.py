from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "src" / "oviqs" / "contracts" / "jsonschema"
OUT_DIR = ROOT / "docs" / "reference" / "schemas"

SCHEMA_EXAMPLES: dict[str, dict[str, Any]] = {
    "eval_request.schema.json": {
        "model": "models/tiny-ov",
        "dataset": "data/likelihood.jsonl",
        "out": "reports/current.json",
        "backend": "openvino-runtime",
        "device": "GPU",
        "window_size": 4096,
        "stride": 1024,
        "labels": {"profile": "docs-smoke"},
    },
    "evaluation_report.schema.json": {
        "schema_version": "openvino_llm_quality_v1",
        "run": {
            "id": "docs-gpu-smoke",
            "suite": "openvino_llm_quality_v1",
            "model": "models/tiny-ov",
            "device": "GPU",
            "precision": "FP16",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        "summary": {
            "overall_status": "warning",
            "main_findings": ["GPU run completed with one unsupported evidence path."],
        },
        "metric_references": {
            "likelihood.perplexity": {"oracle": "lower_is_better", "target": 3.0}
        },
    },
    "metric_observation.schema.json": {
        "path": "likelihood.perplexity",
        "section": "likelihood",
        "name": "perplexity",
        "value": 2.71,
        "unit": None,
        "status": "pass",
        "severity": "none",
    },
    "report_bundle.schema.json": {
        "created_at": "2026-01-01T00:00:00+00:00",
        "schema_version": "openvino_llm_quality_v1",
        "source_report": "report.json",
        "baseline_report": "baseline.json",
    },
    "sample_metric.schema.json": {
        "section": "likelihood",
        "sample_index": 0,
        "sample_id": "prompt-0001",
        "metric": "negative_log_likelihood",
        "value": 1.24,
        "status": "pass",
    },
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expected_pages: set[Path] = set()
    for schema_path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        slug = schema_path.name.removesuffix(".schema.json").replace("_", "-")
        target = OUT_DIR / f"{slug}.md"
        target.write_text(render_schema_page(schema_path, schema), encoding="utf-8")
        expected_pages.add(target)
    for stale_page in OUT_DIR.glob("*.md"):
        if stale_page not in expected_pages:
            stale_page.unlink()


def render_schema_page(schema_path: Path, schema: dict[str, Any]) -> str:
    required = schema.get("required", [])
    lines = [
        f"# {schema.get('title', schema_path.stem)}",
        "",
        "Generated from the repository JSON Schema contract.",
        "",
        f"- Raw schema: `{schema_path.relative_to(ROOT)}`",
        f"- `$id`: `{schema.get('$id', '')}`",
        f"- Required fields: {', '.join(f'`{item}`' for item in required) or 'none'}",
        "",
        "## Fields",
        "",
        "| Field | Type | Required | Description |",
        "|---|---|---|---|",
    ]
    properties = schema.get("properties", {})
    for name, spec in sorted(properties.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_table_cell(f"`{name}`"),
                    markdown_table_cell(f"`{format_type(spec)}`"),
                    "yes" if name in required else "no",
                    markdown_table_cell(str(spec.get("description", ""))),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Compatibility",
            "",
            "Schemas are additive before the first stable release. Missing or unsupported metrics "
            "must be represented as `unknown` instead of approximated values.",
            "",
            "Required fields are the minimum portable contract. Producers may include additional "
            "fields while OVIQS is pre-stable, but consumers should ignore unknown keys unless a "
            "documented schema version says otherwise.",
            "",
            "## Status and enum values",
            "",
            *format_enum_notes(schema),
            "",
            "## Example",
            "",
            "```json",
            json.dumps(SCHEMA_EXAMPLES.get(schema_path.name, {}), indent=2, sort_keys=True),
            "```",
            "",
            "## Produced by",
            "",
            "- `oviq eval-*` commands",
            "- `oviq run-suite` and `oviq run-gpu-suite`",
            "- `oviq report build` for bundle metadata and normalized metrics",
            "",
        ]
    )
    return "\n".join(lines)


def format_type(spec: dict[str, Any]) -> str:
    value = spec.get("type", "object")
    return " | ".join(value) if isinstance(value, list) else str(value)


def markdown_table_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def format_enum_notes(schema: dict[str, Any]) -> list[str]:
    enums = collect_enums(schema)
    if not enums:
        return ["This schema does not currently constrain fields with enum values."]
    return [
        f"- `{path}` allows {', '.join(f'`{value}`' for value in values)}."
        for path, values in enums
    ]


def collect_enums(schema: dict[str, Any], prefix: str = "$") -> list[tuple[str, list[str]]]:
    enums: list[tuple[str, list[str]]] = []
    if "enum" in schema:
        enums.append((prefix, [str(value) for value in schema["enum"]]))
    for name, spec in sorted(schema.get("properties", {}).items()):
        if isinstance(spec, dict):
            enums.extend(collect_enums(spec, f"{prefix}.{name}"))
    items = schema.get("items")
    if isinstance(items, dict):
        enums.extend(collect_enums(items, f"{prefix}[]"))
    return enums


if __name__ == "__main__":
    main()
