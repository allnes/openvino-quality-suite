# `oviq list-genai-models`

Generated from the Typer command help.

```text

 Usage: oviq list-genai-models [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --config        PATH  GenAI model matrix YAML                                │
│                       [default: configs/examples/genai_metric_models.yaml]   │
│ --tier          TEXT  Filter by tier                                         │
│ --metric        TEXT  Filter by metric                                       │
│ --family        TEXT  Filter by family                                       │
│ --json                Print JSON                                             │
│ --help                Show this message and exit.                            │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Notes

This command reports the model catalogue known to the current checkout. Treat it as a planning aid, then validate the exact exported model directory with a real evaluation before publishing a quality result.
