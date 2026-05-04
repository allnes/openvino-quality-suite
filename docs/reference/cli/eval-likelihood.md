# `oviq eval-likelihood`

Generated from the Typer command help.

```text

 Usage: oviq eval-likelihood [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --model              TEXT     Model path or id [required]                 │
│ *  --dataset            PATH     JSONL dataset [required]                    │
│ *  --out                PATH     Output JSON report [required]               │
│    --backend            TEXT     dummy, hf, optimum-openvino,                │
│                                  openvino-runtime                            │
│                                  [default: dummy]                            │
│    --device             TEXT     Inference device [default: CPU]             │
│    --window-size        INTEGER  Sliding window size [default: 4096]         │
│    --stride             INTEGER  Sliding window stride [default: 1024]       │
│    --help                        Show this message and exit.                 │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq eval-likelihood \
  --model models/tiny-ov \
  --dataset data/likelihood.jsonl \
  --out reports/likelihood.json \
  --backend openvino-runtime \
  --device CPU
```

Use this command when aligned logits are available. If the selected backend cannot return token-level likelihood evidence, the affected metrics must stay `unknown` instead of being approximated from generated text.
