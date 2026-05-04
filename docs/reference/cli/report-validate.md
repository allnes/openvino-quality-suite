# `oviq report validate`

Generated from the Typer command help.

```text

 Usage: oviq report validate [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --report        PATH  Input EvaluationReport JSON [required]              │
│    --help                Show this message and exit.                         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq report validate --report reports/current.json
```

Validation checks the public report contract. It does not prove that every metric is meaningful; unsupported evidence should still be represented as `unknown`.
