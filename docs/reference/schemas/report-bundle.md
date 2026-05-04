# OVIQS Report Bundle

Generated from the repository JSON Schema contract.

- Raw schema: `src/oviqs/contracts/jsonschema/report_bundle.schema.json`
- `$id`: `https://github.com/allnes/openvino-quality-suite/blob/main/src/oviqs/contracts/jsonschema/report_bundle.schema.json`
- Required fields: `root`, `report_json`, `analysis_json`, `metrics_csv`, `sample_metrics_jsonl`, `index_md`, `dashboard_html`

## Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `analysis_json` | `string` | yes |  |
| `dashboard_html` | `string` | yes |  |
| `index_md` | `string` | yes |  |
| `metadata_json` | `string \| null` | no |  |
| `metrics_csv` | `string` | yes |  |
| `report_json` | `string` | yes |  |
| `root` | `string` | yes |  |
| `sample_metrics_jsonl` | `string` | yes |  |

## Compatibility

Schemas are additive before the first stable release. Missing or unsupported metrics must be represented as `unknown` instead of approximated values.

Required fields are the minimum portable contract. Producers may include additional fields while OVIQS is pre-stable, but consumers should ignore unknown keys unless a documented schema version says otherwise.

## Status and enum values

This schema does not currently constrain fields with enum values.

## Example

```json
{
  "baseline_report": "baseline.json",
  "created_at": "2026-01-01T00:00:00+00:00",
  "schema_version": "openvino_llm_quality_v1",
  "source_report": "report.json"
}
```

## Produced by

- `oviq eval-*` commands
- `oviq run-suite` and `oviq run-gpu-suite`
- `oviq report build` for bundle metadata and normalized metrics
