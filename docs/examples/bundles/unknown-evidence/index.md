# Unknown evidence report

Fixture showing how unsupported evidence is preserved as unknown.

## Reproduce the scenario

```bash
oviq eval-likelihood \
  --model models/generation-only \
  --backend ov-genai \
  --dataset data/likelihood.jsonl \
  --out report.json
```

The command documents the intended workflow for this fixture. The generated files in this directory are deterministic examples, not benchmark claims.

## Bundle contents

- Run id: `docs-unknown-evidence`
- Overall status: `unknown`
- Files: `report.json`, `analysis.json`, `metrics.csv`, `sample_metrics.jsonl`, `metadata.json`, `dashboard.html`

## Metric rows

| Path | Section | Name | Status |
|---|---|---|---|
| `likelihood.status` | `likelihood` | `status` | `unknown` |

## Review notes

- The report has no substituted likelihood value because aligned logits are absent.
- `analysis.json` explains the unsupported evidence path instead of marking it pass.
- Use this bundle when testing mandatory gates over missing evidence.

Use this fixture when checking documentation examples, schema validation, renderer behavior and reporting copy without relying on external model downloads.
