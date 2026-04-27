# Data Formats

OVIQS uses JSONL for sample-oriented inputs and JSON for reports.

## Evaluation samples

Each line is an object validated as `EvalSample`:

```json
{"id":"s1","task_type":"likelihood","text":"The model input text."}
```

Fields:

- `id`: stable sample identifier.
- `task_type`: one of `likelihood`, `long_context`, `controlled_context`, `rag`,
  `agent`, `generation` or `serving`.
- `text`: full text for likelihood-style evaluation.
- `prompt`, `context`, `target`: structured prompt fields.
- `references`: accepted reference answers.
- `retrieved_contexts`: RAG contexts.
- `expected_evidence`: evidence strings expected in retrieved or generated content.
- `expected_constraints`: task-specific constraints for RAG or agent evaluation.
- `metadata`: free-form metadata.

Validation rules:

- `likelihood` requires `text` or `prompt`.
- `long_context` and `controlled_context` require `target`.

## Tokenized samples

`TokenizedSample` is available when callers want to bypass text tokenization:

```json
{
  "id": "s1",
  "input_ids": [1, 42, 2],
  "attention_mask": [1, 1, 1],
  "target_mask": [0, 1, 1],
  "metadata": {}
}
```

## Agent traces

Agent traces are ordered event sequences:

```json
{
  "id": "t1",
  "input": "Find the current answer using tools.",
  "steps": [
    {"type": "message", "role": "user", "content": "Find the answer."},
    {"type": "tool_call", "tool": "search", "args": {"q": "example"}},
    {"type": "observation", "result": {"status": "ok"}},
    {"type": "final", "content": "Answer"}
  ],
  "expected_tools": [],
  "expected_state": {},
  "expected_constraints": {},
  "metadata": {}
}
```

Step `type` must be one of `message`, `tool_call`, `observation`, `final` or `error`.
