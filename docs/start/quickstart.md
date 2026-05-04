# Quickstart

This path produces one schema-valid report and one reviewable report bundle
without downloading a model.

## 1. Install

From the repository root:

```bash
uv sync --extra dev
```

If you are not using `uv`, install the package in editable mode:

```bash
python -m pip install -e ".[dev]"
```

## 2. Create a tiny dataset

```bash
printf '{"id":"s1","task_type":"likelihood","text":"this is a test"}\n' > /tmp/likelihood.jsonl
```

The likelihood command expects JSONL rows with at least `id`, `task_type` and
`text` for this minimal case.

## 3. Run the dummy backend

```bash
oviq eval-likelihood \
  --model dummy \
  --backend dummy \
  --dataset /tmp/likelihood.jsonl \
  --out /tmp/likelihood.json
```

The dummy backend is deterministic and local. It is intended for smoke tests,
documentation examples and CI checks of the reporting path.

## 4. Validate and package the report

```bash
oviq report validate --report /tmp/likelihood.json

oviq report build \
  --report /tmp/likelihood.json \
  --out /tmp/likelihood-report \
  --format all
```

The generated bundle includes:

- `report.json`: canonical machine-readable report.
- `analysis.json`: normalized metrics and findings.
- `metrics.csv`: flat scalar metric table.
- `index.md`: Markdown summary for review.
- `dashboard.html`: self-contained HTML dashboard.
- `metadata.json`: bundle provenance.
- `assets/`: static bundle assets.

## 5. Render another view

```bash
oviq report render \
  --bundle /tmp/likelihood-report \
  --format markdown \
  --out /tmp/likelihood.md
```

Open `/tmp/likelihood-report/index.md` or
`/tmp/likelihood-report/dashboard.html` to inspect the result.

## Next steps

- Compare against a baseline with
  [Compare against a baseline](../tutorials/compare-against-a-baseline.md).
- Learn the report files in
  [Bundle layout](../reference/reporting/bundle-layout.md).
- Add CI validation with [Integrate in CI](../how-to/integrate-in-ci.md).
