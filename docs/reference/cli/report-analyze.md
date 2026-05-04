# `oviq report analyze`

Generated from the Typer command help.

```text

 Usage: oviq report analyze [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --report               PATH  Input EvaluationReport JSON [required]       │
│ *  --out                  PATH  Output analysis JSON [required]              │
│    --baseline             PATH  Optional baseline report JSON                │
│    --gates                PATH  Optional gates YAML                          │
│    --trend-history        PATH  Optional report-history JSONL used as trend  │
│                                 baseline                                     │
│    --help                       Show this message and exit.                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq report analyze \
  --report reports/current.json \
  --baseline reports/baseline.json \
  --gates configs/gates.yaml \
  --out reports/current.analysis.json
```

Analysis output is machine-readable. Use it in CI when the rendered dashboard is too heavy and the job only needs findings, gate results, and metric status.
