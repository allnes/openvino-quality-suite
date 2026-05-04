# Reference comparison report

Two-run comparison fixture with a baseline report reference.

## Reproduce the scenario

```bash
oviq report reference-comparison \
  --report baseline=baseline.json \
  --report current=report.json \
  --out comparison.md
```

The command documents the intended workflow for this fixture. The generated files in this directory are deterministic examples, not benchmark claims.

## Bundle contents

- Run id: `docs-reference-current`
- Overall status: `pass`
- Files: `report.json`, `analysis.json`, `metrics.csv`, `sample_metrics.jsonl`, `metadata.json`, `dashboard.html`

## Metric rows

| Path | Section | Name | Status |
|---|---|---|---|
| `likelihood.perplexity` | `likelihood` | `perplexity` | `pass` |
| `likelihood.nll` | `likelihood` | `nll` | `pass` |

## Review notes

- `run.reference` and `run.current` identify the compared runs.
- `analysis.json` includes a passing finding for `likelihood.perplexity`.
- Use this bundle when checking comparison tables and baseline labels.

Use this fixture when checking documentation examples, schema validation, renderer behavior and reporting copy without relying on external model downloads.
