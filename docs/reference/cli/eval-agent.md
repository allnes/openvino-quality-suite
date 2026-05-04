# `oviq eval-agent`

Generated from the Typer command help.

```text

 Usage: oviq eval-agent [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --traces          PATH  Agent traces JSONL [required]                     │
│ *  --out             PATH  Output JSON report [required]                     │
│    --expected        PATH  Expected JSONL                                    │
│    --help                  Show this message and exit.                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq eval-agent \
  --traces data/agent-traces.jsonl \
  --expected data/agent-expected.jsonl \
  --out reports/agent.json
```

Agent checks depend on trace structure. Keep tool calls, observations, final answers and expected state stable in fixtures so failures can be debugged from the report bundle without replaying a live agent.
