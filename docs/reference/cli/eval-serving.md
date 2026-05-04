# `oviq eval-serving`

Generated from the Typer command help.

```text

 Usage: oviq eval-serving [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --out            PATH  Output JSON report [required]                      │
│    --model          TEXT  Production model path/id [default: dummy]          │
│    --dataset        PATH  Serving JSONL dataset                              │
│    --backend        TEXT  Logits backend [default: dummy]                    │
│    --device         TEXT  Device [default: CPU]                              │
│    --help                 Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq eval-serving \
  --model ovms://localhost:9000/models/tiny \
  --dataset data/serving.jsonl \
  --backend dummy \
  --device CPU \
  --out reports/serving.json
```

Serving diagnostics are for endpoint behavior such as batch invariance, prefix stability and cache-sensitive checks. Keep endpoint metadata in the report reproducibility section when publishing results.
