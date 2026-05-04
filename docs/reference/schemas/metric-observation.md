# OVIQS Metric Observation

Generated from the repository JSON Schema contract.

- Raw schema: `src/oviqs/contracts/jsonschema/metric_observation.schema.json`
- `$id`: `https://github.com/allnes/openvino-quality-suite/blob/main/src/oviqs/contracts/jsonschema/metric_observation.schema.json`
- Required fields: `path`, `section`, `name`, `status`, `severity`

## Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `baseline_value` | `number \| integer \| null` | no |  |
| `degradation_rule` | `string \| null` | no |  |
| `delta_abs` | `number \| null` | no |  |
| `delta_rel` | `number \| null` | no |  |
| `name` | `string` | yes |  |
| `path` | `string` | yes |  |
| `reference_id` | `string \| null` | no |  |
| `sample_count` | `integer \| null` | no |  |
| `section` | `string` | yes |  |
| `severity` | `string` | yes |  |
| `status` | `string` | yes |  |
| `tags` | `array` | no |  |
| `threshold` | `number \| integer \| null` | no |  |
| `threshold_rule` | `string \| null` | no |  |
| `unit` | `string \| null` | no |  |
| `value` | `number \| integer \| string \| boolean \| null` | no |  |

## Compatibility

Schemas are additive before the first stable release. Missing or unsupported metrics must be represented as `unknown` instead of approximated values.

Required fields are the minimum portable contract. Producers may include additional fields while OVIQS is pre-stable, but consumers should ignore unknown keys unless a documented schema version says otherwise.

## Status and enum values

- `$.severity` allows `none`, `low`, `medium`, `high`, `critical`.
- `$.status` allows `pass`, `warning`, `fail`, `unknown`.

## Example

```json
{
  "name": "perplexity",
  "path": "likelihood.perplexity",
  "section": "likelihood",
  "severity": "none",
  "status": "pass",
  "unit": null,
  "value": 2.71
}
```

## Produced by

- `oviq eval-*` commands
- `oviq run-suite` and `oviq run-gpu-suite`
- `oviq report build` for bundle metadata and normalized metrics
