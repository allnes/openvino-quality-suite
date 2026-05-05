# `oviq report reference-comparison`

Generated from the Typer command help.

```text

 Usage: oviq report reference-comparison [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --report             TEXT  Report JSON path, optionally LABEL=PATH.       │
│                               Repeat for multiple reports.                   │
│                               [required]                                     │
│ *  --out                PATH  Output comparison table [required]             │
│    --format             TEXT  markdown, markdown-transposed, html-by-model,  │
│                               html-dashboard, csv or json                    │
│                               [default: markdown]                            │
│    --all-metrics              Include every metric listed in report coverage │
│    --help                     Show this message and exit.                    │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq report reference-comparison \
  --report baseline=reports/baseline.json \
  --report current=reports/current.json \
  --out reports/reference-comparison.md \
  --format markdown
```

Use labels when comparing multiple reports. The command lives under `oviq report`; legacy top-level comparison commands are intentionally not part of the new CLI.
