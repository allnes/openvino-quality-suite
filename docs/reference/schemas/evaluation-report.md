# OVIQS Evaluation Report

Generated from the repository JSON Schema contract.

- Raw schema: `src/oviqs/contracts/jsonschema/evaluation_report.schema.json`
- `$id`: `https://github.com/allnes/openvino-quality-suite/blob/main/src/oviqs/contracts/jsonschema/evaluation_report.schema.json`
- Required fields: `schema_version`, `run`, `summary`

## Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `analysis` | `object` | no |  |
| `artifacts` | `object` | no |  |
| `metric_references` | `object` | no |  |
| `raw_sample_metrics_uri` | `string \| null` | no |  |
| `report_metadata` | `object` | no |  |
| `run` | `object` | yes |  |
| `sample_metrics_summary` | `object` | no |  |
| `schema_version` | `string` | yes |  |
| `summary` | `object` | yes |  |
| `ui_hints` | `object` | no |  |

## Compatibility

Schemas are additive before the first stable release. Missing or unsupported metrics must be represented as `unknown` instead of approximated values.

Required fields are the minimum portable contract. Producers may include additional fields while OVIQS is pre-stable, but consumers should ignore unknown keys unless a documented schema version says otherwise.

## Status and enum values

- `$.analysis.findings[].severity` allows `none`, `low`, `medium`, `high`, `critical`.
- `$.analysis.findings[].status` allows `pass`, `warning`, `fail`, `unknown`.
- `$.analysis.metrics[].severity` allows `none`, `low`, `medium`, `high`, `critical`.
- `$.analysis.metrics[].status` allows `pass`, `warning`, `fail`, `unknown`.
- `$.summary.overall_status` allows `pass`, `warning`, `fail`, `unknown`.

## Example

```json
{
  "metric_references": {
    "likelihood.perplexity": {
      "oracle": "lower_is_better",
      "target": 3.0
    }
  },
  "run": {
    "created_at": "2026-01-01T00:00:00+00:00",
    "device": "GPU",
    "id": "docs-gpu-smoke",
    "model": "models/tiny-ov",
    "precision": "FP16",
    "suite": "openvino_llm_quality_v1"
  },
  "schema_version": "openvino_llm_quality_v1",
  "summary": {
    "main_findings": [
      "GPU run completed with one unsupported evidence path."
    ],
    "overall_status": "warning"
  }
}
```

## Produced by

- `oviq eval-*` commands
- `oviq run-suite` and `oviq run-gpu-suite`
- `oviq report build` for bundle metadata and normalized metrics
