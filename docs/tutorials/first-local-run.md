# First local run

Goal: produce one schema-valid report, inspect it, and leave behind a bundle
that can be uploaded as a CI artifact.

## Prerequisites

Install the development environment:

```bash
uv sync --extra dev
```

No model download is required for this tutorial.

## Create a fixture

```bash
mkdir -p /tmp/oviqs-demo
printf '{"id":"s1","task_type":"likelihood","text":"this is a test"}\n' \
  > /tmp/oviqs-demo/likelihood.jsonl
```

The dummy backend reads the same JSONL shape as the likelihood evaluator. That
keeps the first run representative without depending on OpenVINO or Hugging Face
runtime packages.

## Run the evaluator

```bash
oviq eval-likelihood \
  --model dummy \
  --backend dummy \
  --dataset /tmp/oviqs-demo/likelihood.jsonl \
  --out /tmp/oviqs-demo/likelihood.json
```

The output is an `EvaluationReport`. It includes a `schema_version`, `run`
metadata, a `summary`, and a `likelihood` section.

## Validate and bundle the output

```bash
oviq report validate --report /tmp/oviqs-demo/likelihood.json

oviq report build \
  --report /tmp/oviqs-demo/likelihood.json \
  --out /tmp/oviqs-demo/bundle \
  --format all
```

Inspect the bundle tree:

```bash
find /tmp/oviqs-demo/bundle -maxdepth 2 -type f | sort
```

Expected files include `report.json`, `analysis.json`, `metrics.csv`,
`index.md`, `dashboard.html` and `metadata.json`.

## What to check

- `report.json` is the canonical machine-readable artifact.
- `analysis.json` contains normalized observations and findings.
- `metrics.csv` is useful for CI tables and quick diffs.
- `index.md` is the pull-request friendly summary.
- `dashboard.html` is a self-contained reviewer view.

## Cleanup

```bash
rm -rf /tmp/oviqs-demo
```
