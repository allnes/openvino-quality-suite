# `oviq eval-rag`

Generated from the Typer command help.

```text

 Usage: oviq eval-rag [OPTIONS]

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --dataset        PATH  RAG JSONL dataset [required]                       │
│ *  --out            PATH  Output JSON report [required]                      │
│    --answers        PATH  Answers JSONL                                      │
│    --scorer         TEXT  placeholder or ragas [default: placeholder]        │
│    --help                 Show this message and exit.                        │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Example

```bash
oviq eval-rag \
  --dataset data/rag-questions.jsonl \
  --answers reports/rag-answers.jsonl \
  --scorer placeholder \
  --out reports/rag.json
```

Use the placeholder scorer for deterministic fixture checks. Use optional RAG scorers only when their dependencies and judge inputs are available, and keep judge-based gaps visible as `unknown`.
