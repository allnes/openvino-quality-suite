# `oviq report build-suite`

Generated from the Typer command help.

```text

 Usage: oviq report build-suite [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --report        TEXT  Report JSON path, optionally LABEL=PATH. Repeat for │
│                          each model.                                         │
│                          [required]                                          │
│ *  --out           PATH  Output suite bundle directory [required]            │
│    --gates         PATH  Optional gates result JSON                          │
│    --title         TEXT  Suite index page title                              │
│                          [default: OVIQS — OpenVINO Inference Quality Suite] │
│    --help                Show this message and exit.                         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq report build-suite \
  --report Qwen3=reports/qwen3.json \
  --report Mistral=reports/mistral.json \
  --report Phi-4=reports/phi4.json \
  --out reports/suite
```

Builds a multi-model suite bundle: a per-model dashboard bundle under `models/<label>/`, a cross-model `comparison.html`, and an `index.html` landing page with a fidelity-first summary table linking them. Use it to publish one navigable report across several exported models.
