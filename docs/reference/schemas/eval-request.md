# OVIQS Evaluation Request

Generated from the repository JSON Schema contract.

- Raw schema: `src/oviqs/contracts/jsonschema/eval_request.schema.json`
- `$id`: `https://github.com/allnes/openvino-quality-suite/blob/main/src/oviqs/contracts/jsonschema/eval_request.schema.json`
- Required fields: `model`, `dataset`, `out`

## Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `backend` | `string` | no |  |
| `dataset` | `string` | yes |  |
| `device` | `string` | no |  |
| `labels` | `object` | no |  |
| `model` | `string` | yes |  |
| `out` | `string` | yes |  |
| `stride` | `integer` | no |  |
| `window_size` | `integer` | no |  |

## Compatibility

Schemas are additive before the first stable release. Missing or unsupported metrics must be represented as `unknown` instead of approximated values.

Required fields are the minimum portable contract. Producers may include additional fields while OVIQS is pre-stable, but consumers should ignore unknown keys unless a documented schema version says otherwise.

## Status and enum values

This schema does not currently constrain fields with enum values.

## Example

```json
{
  "backend": "openvino-runtime",
  "dataset": "data/likelihood.jsonl",
  "device": "GPU",
  "labels": {
    "profile": "docs-smoke"
  },
  "model": "models/tiny-ov",
  "out": "reports/current.json",
  "stride": 1024,
  "window_size": 4096
}
```

## Produced by

- `oviq eval-*` commands
- `oviq run-suite` and `oviq run-gpu-suite`
- `oviq report build` for bundle metadata and normalized metrics
