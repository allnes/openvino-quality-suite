# `oviq metric-long-context`

Generated from the Typer command help.

```text

 Usage: oviq metric-long-context [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --position-ppl-json        TEXT  JSON object with position bucket PPL     │
│                                     [required]                               │
│    --help                           Show this message and exit.              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq metric-long-context \
  --position-ppl-json '{"start":2.1,"middle":3.4,"end":2.3}'
```

This helper is a deterministic metric utility for quick checks and documentation examples. Full long-context reports should still be produced with `oviq eval-long-context`.
