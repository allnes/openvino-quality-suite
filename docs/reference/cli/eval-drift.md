# `oviq eval-drift`

Generated from the Typer command help.

```text

 Usage: oviq eval-drift [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --reference                TEXT  Reference model path/id [required]       │
│ *  --current                  TEXT  Current model path/id [required]         │
│ *  --dataset                  PATH  JSONL dataset [required]                 │
│ *  --out                      PATH  Output JSON report [required]            │
│    --reference-backend        TEXT  Reference backend [default: dummy]       │
│    --current-backend          TEXT  Current backend [default: dummy]         │
│    --reference-device         TEXT  Inference device for reference model     │
│                                     [default: CPU]                           │
│    --device                   TEXT  Inference device for current model       │
│                                     [default: CPU]                           │
│    --help                           Show this message and exit.              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq eval-drift \
  --reference models/baseline-ov \
  --current models/current-ov \
  --dataset data/likelihood.jsonl \
  --out reports/drift.json \
  --reference-backend openvino-runtime \
  --current-backend openvino-runtime \
  --device CPU
```

Use drift checks when reference and current runs can be aligned over the same samples, tokenizer positions and metric definitions. If alignment cannot be proven, report the affected comparison as `unknown`.
