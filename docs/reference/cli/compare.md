# `oviq compare`

Generated from the Typer command help.

```text

 Usage: oviq compare [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --baseline        PATH  Baseline report JSON [required]                   │
│ *  --current         PATH  Current report JSON [required]                    │
│ *  --out             PATH  Output comparison JSON [required]                 │
│    --gates           PATH  Gate YAML                                         │
│    --help                  Show this message and exit.                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq compare \
  --baseline reports/baseline.json \
  --current reports/current.json \
  --gates configs/gates/default_gates.yaml \
  --out reports/comparison.json
```

Use this command for a compact comparison JSON. Prefer `oviq report analyze` or `oviq report build` when the review needs normalized findings, CSV metrics, Markdown or dashboard artifacts.
