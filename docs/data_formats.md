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

## Evaluation reports

Evaluation output is a versioned JSON object with `schema_version`, `run`, `summary` and
diagnostic sections such as `likelihood`, `inference_equivalence`, `long_context`,
`generation`, `rag`, `agent`, `serving`, `performance` and `reproducibility`.

Optional reporting fields are additive:

- `analysis`: normalized metric observations, findings, regressions and outliers.
- `artifacts`: links to generated report files.
- `report_metadata`: builder metadata.
- `ui_hints`: renderer-only hints.
- `sample_metrics_summary`: sample-level summary data.
- `raw_sample_metrics_uri`: external pointer to detailed sample metrics.

`report validate` checks the JSON envelope against the project JSON Schema. It uses
`jsonschema` when available and falls back to the built-in validator in minimal
environments. Known reporting fields such as `analysis.metrics`, `analysis.findings`,
`artifacts`, `report_metadata` and `sample_metrics_summary` have explicit schema shapes
while still allowing additive metric-specific properties.

Report readers normalize persisted reports through
`oviqs.application.reporting.schema_normalization` before analysis or rendering. Persisted
reports must already use the current `openvino_llm_quality_v1` contract; missing, older or
future `schema_version` values are rejected.

## Report bundles

`oviq report build` writes a stable directory layout:

```text
report.json
analysis.json
metrics.csv
sample_metrics.jsonl
index.md
dashboard.html
assets/
metadata.json
```

`metrics.csv` is the normalized scalar metric table. The writer uses `pandas` when
installed and falls back to the Python CSV library. `analysis.json` also carries
`trend_points` when a trend store is configured. `sample_metrics.jsonl` contains
per-sample rows extracted from section-level `samples` arrays. Each row is validated
against `src/oviqs/contracts/jsonschema/sample_metric.schema.json` and bundle metadata
records any validation errors.
