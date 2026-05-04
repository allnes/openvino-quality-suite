# Bundle layout

`oviq report build` writes a stable bundle:

```text
report-bundle/
  report.json
  analysis.json
  metrics.csv
  sample_metrics.jsonl
  index.md
  dashboard.html
  assets/
  metadata.json
```

`report.json` is the canonical machine-readable artifact. Markdown and HTML are
rendered views.

## File responsibilities

| File | Purpose | Producer |
|---|---|---|
| `report.json` | Canonical `EvaluationReport` JSON. | Evaluation command or bundle copy. |
| `analysis.json` | Normalized metrics, findings and optional comparison data. | `oviq report build` or `oviq report analyze`. |
| `metrics.csv` | Flat scalar metric table for review and ingestion. | `oviq report metrics-table` or bundle build. |
| `sample_metrics.jsonl` | Optional sample-level metrics. | Evaluation-specific producers. |
| `index.md` | Markdown review summary. | Markdown renderer. |
| `dashboard.html` | Self-contained dashboard. | HTML renderer. |
| `metadata.json` | Bundle provenance. | Bundle build. |
| `assets/` | Static files used by rendered views. | Bundle build or renderers. |

## Stability rule

Tools may rely on the file names above. New optional files can be added, but
existing names should not be repurposed.

## Minimal review sequence

1. Validate `report.json`.
2. Inspect `analysis.json` for findings and metric count.
3. Review `index.md` for the top summary.
4. Open `dashboard.html` for the detailed view.
5. Use `metrics.csv` when a tabular diff is needed.

## Generated examples

The docs generator writes schema-valid fixtures under `docs/examples/bundles/`:

| Bundle | Purpose |
|---|---|
| `minimal-likelihood` | Small report with unknown likelihood evidence. |
| `reference-comparison` | Passing current-vs-baseline comparison fixture. |
| `gpu-suite-smoke` | GPU run with mixed Runtime and GenAI evidence. |
| `gated-regression` | Failing bundle with gate-style analysis output. |
| `unknown-evidence` | Report that preserves unsupported evidence as `unknown`. |

Use these fixtures for documentation examples and renderer tests. They are not
benchmark results and should not be used as model quality claims.
