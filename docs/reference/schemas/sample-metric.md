# OVIQS Sample Metric Row

Generated from the repository JSON Schema contract.

- Raw schema: `src/oviqs/contracts/jsonschema/sample_metric.schema.json`
- `$id`: `https://github.com/allnes/openvino-quality-suite/blob/main/src/oviqs/contracts/jsonschema/sample_metric.schema.json`
- Required fields: `section`, `sample_index`

## Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | `string \| integer \| null` | no |  |
| `sample_id` | `string \| integer \| null` | no |  |
| `sample_index` | `integer` | yes |  |
| `section` | `string` | yes |  |
| `status` | `string` | no |  |

## Compatibility

Schemas are additive before the first stable release. Missing or unsupported metrics must be represented as `unknown` instead of approximated values.

Required fields are the minimum portable contract. Producers may include additional fields while OVIQS is pre-stable, but consumers should ignore unknown keys unless a documented schema version says otherwise.

## Status and enum values

- `$.status` allows `pass`, `warning`, `fail`, `unknown`.

## Example

```json
{
  "metric": "negative_log_likelihood",
  "sample_id": "prompt-0001",
  "sample_index": 0,
  "section": "likelihood",
  "status": "pass",
  "value": 1.24
}
```

## Produced by

- `oviq eval-*` commands
- `oviq run-suite` and `oviq run-gpu-suite`
- `oviq report build` for bundle metadata and normalized metrics
