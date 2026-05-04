# Gated regression report

Failing bundle that demonstrates gate output and severity.

## Reproduce the scenario

```bash
oviq report analyze \
  --report report.json \
  --baseline baseline.json \
  --gates gates.yaml \
  --out analysis.json
```

The command documents the intended workflow for this fixture. The generated files in this directory are deterministic examples, not benchmark claims.

## Bundle contents

- Run id: `docs-gated-regression`
- Overall status: `fail`
- Files: `report.json`, `analysis.json`, `metrics.csv`, `sample_metrics.jsonl`, `metadata.json`, `dashboard.html`

## Metric rows

| Path | Section | Name | Status |
|---|---|---|---|
| `likelihood.perplexity` | `likelihood` | `perplexity` | `fail` |
| `likelihood.nll` | `likelihood` | `nll` | `warning` |

## Review notes

- `summary.overall_status` is `fail` and the primary finding has high severity.
- `metrics.csv` includes both failing and warning likelihood rows.
- Use this bundle when validating dashboards, gate copy and CI failure summaries.

Use this fixture when checking documentation examples, schema validation, renderer behavior and reporting copy without relying on external model downloads.
