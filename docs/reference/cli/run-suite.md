# `oviq run-suite`

Generated from the Typer command help.

```text

 Usage: oviq run-suite [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --config        PATH  Suite YAML config [required]                        │
│ *  --out           PATH  Output JSON report [required]                       │
│    --help                Show this message and exit.                         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq run-suite --config suites/local.yaml --out reports/current.json
```

The suite command is for orchestrated runs. Keep per-command experiments in the `eval-*` commands until their inputs and expected evidence are stable enough for a checked-in suite profile.
