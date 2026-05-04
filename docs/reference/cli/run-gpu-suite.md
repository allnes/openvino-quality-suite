# `oviq run-gpu-suite`

Generated from the Typer command help.

```text

 Usage: oviq run-gpu-suite [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --model              TEXT     OpenVINO eval/logits model directory        │
│                                  [required]                                  │
│ *  --dataset            PATH     Likelihood JSONL dataset [required]         │
│ *  --out                PATH     Output JSON report [required]               │
│    --backend            TEXT     Logits backend [default: openvino-runtime]  │
│    --device             TEXT     OpenVINO device for metric verification     │
│                                  [default: GPU]                              │
│    --window-size        INTEGER  Long-context sliding window [default: 64]   │
│    --stride             INTEGER  Long-context sliding stride [default: 32]   │
│    --genai-model        TEXT     Optional OpenVINO GenAI model directory for │
│                                  generation layer                            │
│    --help                        Show this message and exit.                 │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq run-gpu-suite \
  --model models/tiny-ov \
  --genai-model models/tiny-genai-ov \
  --dataset data/likelihood.jsonl \
  --out reports/gpu.json \
  --device GPU
```

Use `--model` for Runtime-backed logits checks and `--genai-model` for generation checks. A valid GPU run may still produce `unknown` metrics when a backend cannot expose the required evidence.
