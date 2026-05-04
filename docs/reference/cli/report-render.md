# `oviq report render`

Generated from the Typer command help.

```text

 Usage: oviq report render [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --bundle        PATH  Report bundle directory [required]                  │
│ *  --out           PATH  Output rendered artifact [required]                 │
│    --format        TEXT  markdown or html-dashboard [default: markdown]      │
│    --help                Show this message and exit.                         │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq report render \
  --bundle reports/current-bundle \
  --out reports/current-bundle/index.md \
  --format markdown
```

Render from an existing bundle so reports, metrics, analysis, and metadata stay consistent across Markdown and dashboard outputs.
