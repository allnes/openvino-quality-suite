# `oviq eval-long-context`

Generated from the Typer command help.

```text

 Usage: oviq eval-long-context [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --dataset            PATH     JSONL dataset or precomputed metrics        │
│                                  [required]                                  │
│ *  --out                PATH     Output JSON report [required]               │
│    --model              TEXT     Model path or id [default: dummy]           │
│    --backend            TEXT     Backend [default: dummy]                    │
│    --device             TEXT     Inference device [default: CPU]             │
│    --lengths            TEXT     Comma-separated context lengths             │
│                                  [default: 4096,8192,16384]                  │
│    --window-size        INTEGER  Sliding-window PPL window [default: 4096]   │
│    --stride             INTEGER  Sliding-window PPL stride [default: 1024]   │
│    --help                        Show this message and exit.                 │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq eval-long-context \
  --dataset data/long-context.jsonl \
  --model models/tiny-ov \
  --backend openvino-runtime \
  --lengths 4096,8192,16384 \
  --out reports/long-context.json
```

Keep the length set explicit so repeated CI runs compare the same context windows. Position and distractor metrics should stay separate from generic generation quality scores.
