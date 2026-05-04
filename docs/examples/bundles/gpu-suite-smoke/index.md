# GPU suite smoke report

GPU fixture with Runtime and GenAI evidence recorded in one report.

## Reproduce the scenario

```bash
oviq run-gpu-suite \
  --model models/tiny-ov \
  --genai-model models/tiny-genai-ov \
  --dataset data/likelihood.jsonl \
  --out report.json \
  --device GPU
```

The command documents the intended workflow for this fixture. The generated files in this directory are deterministic examples, not benchmark claims.

## Bundle contents

- Run id: `docs-gpu-suite-smoke`
- Overall status: `warning`
- Files: `report.json`, `analysis.json`, `metrics.csv`, `sample_metrics.jsonl`, `metadata.json`, `dashboard.html`

## Metric rows

| Path | Section | Name | Status |
|---|---|---|---|
| `likelihood.perplexity` | `likelihood` | `perplexity` | `warning` |
| `likelihood.generation.status` | `likelihood` | `generation.status` | `unknown` |

## Review notes

- `run.device` is `GPU` so device-sensitive reporting paths can be tested.
- `likelihood.generation.status` remains `unknown` when GenAI logits are unavailable.
- Use this bundle for GPU docs without requiring a real device in documentation tests.

Use this fixture when checking documentation examples, schema validation, renderer behavior and reporting copy without relying on external model downloads.
