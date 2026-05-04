# `oviq genai-export-plan`

Generated from the Typer command help.

```text

 Usage: oviq genai-export-plan [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --model              TEXT  Model id from the GenAI model matrix           │
│                               [required]                                     │
│    --config             PATH  GenAI model matrix YAML                        │
│                               [default:                                      │
│                               configs/examples/genai_metric_models.yaml]     │
│    --output-root        PATH  Root directory for exported models             │
│                               [default: models]                              │
│    --variant            TEXT  Export variant; repeat option for multiple     │
│                               variants                                       │
│    --json                     Print JSON                                     │
│    --help                     Show this message and exit.                    │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq genai-export-plan --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

The export plan is documentation for the operator. It does not replace an actual OpenVINO model export or a subsequent `oviq run-gpu-suite` validation run.
