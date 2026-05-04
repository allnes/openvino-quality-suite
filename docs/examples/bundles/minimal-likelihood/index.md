# Minimal likelihood report

Smallest useful likelihood report with unknown evidence status.

## Reproduce the scenario

```bash
oviq eval-likelihood \
  --model dummy \
  --backend dummy \
  --dataset data/likelihood.jsonl \
  --out report.json
```

The command documents the intended workflow for this fixture. The generated files in this directory are deterministic examples, not benchmark claims.

## Bundle contents

- Run id: `docs-minimal-likelihood`
- Overall status: `unknown`
- Files: `report.json`, `analysis.json`, `metrics.csv`, `sample_metrics.jsonl`, `metadata.json`, `dashboard.html`

## Metric rows

| Path | Section | Name | Status |
|---|---|---|---|
| `likelihood.perplexity` | `likelihood` | `perplexity` | `unknown` |

## Review notes

- `summary.overall_status` is `unknown` because this fixture demonstrates missing evidence.
- `metrics.csv` contains one scalar path: `likelihood.perplexity`.
- Use this bundle as the smallest schema-validation and renderer smoke fixture.

Use this fixture when checking documentation examples, schema validation, renderer behavior and reporting copy without relying on external model downloads.
