# `oviq list-metric-references`

Generated from the Typer command help.

```text

 Usage: oviq list-metric-references [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --family        TEXT  Filter by metric family                                │
│ --json                Print JSON                                             │
│ --help                Show this message and exit.                            │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq list-metric-references --family likelihood --json
```

Reference metadata tells reviewers which metrics are gateable and which evidence is required. Do not add gates for metrics that do not have a documented reference or oracle.
