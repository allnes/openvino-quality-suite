# `oviq report build`

Generated from the Typer command help.

```text

 Usage: oviq report build [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --report               PATH  Input EvaluationReport JSON [required]       │
│ *  --out                  PATH  Output report bundle directory [required]    │
│    --baseline             PATH  Optional baseline report JSON                │
│    --gates                PATH  Optional gates result JSON                   │
│    --trend-history        PATH  Optional report-history JSONL used as trend  │
│                                 baseline                                     │
│    --format               TEXT  all [default: all]                           │
│    --help                       Show this message and exit.                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq report build \
  --report reports/current.json \
  --baseline reports/baseline.json \
  --gates configs/gates.yaml \
  --out reports/current-bundle \
  --format all
```

The bundle is the publishable reporting artifact. It should contain the source report, normalized metrics, analysis output, metadata, and requested renderings.
