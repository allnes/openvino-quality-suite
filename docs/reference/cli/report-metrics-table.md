# `oviq report metrics-table`

Generated from the Typer command help.

```text

 Usage: oviq report metrics-table [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --report               PATH  Input EvaluationReport JSON [required]       │
│ *  --out                  PATH  Output metrics CSV [required]                │
│    --trend-history        PATH  Optional report-history JSONL used as trend  │
│                                 baseline                                     │
│    --help                       Show this message and exit.                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq report metrics-table \
  --report reports/current.json \
  --out reports/current.metrics.csv
```

Use the CSV for compact review, trend ingestion and spreadsheet-style diffs. Regenerate it from `report.json`; do not edit generated metric rows by hand.
