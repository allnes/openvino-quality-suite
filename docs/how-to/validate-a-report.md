# Validate a report

Validate public reports against the JSON Schema contract:

```bash
oviq report validate --report reports/current.json
```

Validation proves the envelope shape. It does not prove that every optional
metric is present; unsupported metrics should remain `unknown`.

## When to validate

Validate after each evaluation command and before publishing a bundle:

```bash
oviq eval-likelihood \
  --model dummy \
  --backend dummy \
  --dataset /tmp/likelihood.jsonl \
  --out reports/current.json

oviq report validate --report reports/current.json
```

Validate the bundled copy as well when a CI job uploads the bundle:

```bash
oviq report validate --report reports/current-bundle/report.json
```

## What validation catches

Schema validation catches missing required fields such as `schema_version`,
`run`, and `summary`. It also catches invalid status values outside `pass`,
`warning`, `fail`, and `unknown`.

It does not judge metric quality. A report with `overall_status: unknown` can be
schema-valid and still require investigation.
