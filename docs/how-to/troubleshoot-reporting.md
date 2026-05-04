# Troubleshoot reporting

Use this page when a report or bundle is generated but cannot be trusted yet.

## Schema validation fails

Run:

```bash
oviq report validate --report reports/current.json
```

Check that the report has `schema_version`, `run`, and `summary`. The `run`
object must include `id`, `suite`, and `created_at`; `summary.overall_status`
must be one of `pass`, `warning`, `fail`, or `unknown`.

If validation fails after a refactor, regenerate schema docs and compare the
failing field against `src/oviqs/contracts/jsonschema/`. Do not patch generated
docs to hide a schema mismatch; change the producer or the schema contract.

## A metric is missing

Missing evidence is not automatically a bug. Some backends cannot expose aligned
logits or sample-level details. In that case, the metric should remain
`unknown`, and the report should explain which evidence is unavailable.

Use:

```bash
oviq report metrics-table \
  --report reports/current.json \
  --out reports/current.metrics.csv
```

Then inspect whether the expected metric path is present.

If the path is absent from `metrics.csv` but present in `report.json`, the
normalization step is the likely issue. If the path is absent from both files,
the evaluation service or backend adapter did not produce the metric.

## A gate is unknown

An `unknown` gate means the check was not proven. Confirm that the metric exists,
that a reference or oracle is registered when required, and that the gate path
matches the normalized metric path.

Gate troubleshooting order:

1. Confirm the metric path in `metrics.csv`.
2. Confirm the gate YAML uses the same path.
3. Confirm the metric status is not already `unknown`.
4. Confirm the comparison baseline exists when the gate is relative.
5. Re-run `oviq report analyze` with the same report, baseline and gates.

Unknown should block publication when the gate is mandatory. It is acceptable in
exploratory runs only when the bundle explains why the evidence is unavailable.

## A bundle is incomplete

Rebuild the bundle with all renderers:

```bash
oviq report build \
  --report reports/current.json \
  --out reports/current-bundle \
  --format all
```

Expected files are `report.json`, `analysis.json`, `metrics.csv`, `index.md`,
`dashboard.html`, `metadata.json`, and `assets/`.

If `report.json` exists but rendered files are missing, run `oviq report render`
against the existing bundle rather than re-running the evaluation. If
`analysis.json` is missing, rebuild the bundle because renderers depend on the
normalized analysis payload.

## A reference comparison looks wrong

Use explicit labels so the comparison table cannot depend on path names:

```bash
oviq report reference-comparison \
  --report baseline=reports/baseline.json \
  --report current=reports/current.json \
  --out reports/reference-comparison.md \
  --format markdown
```

If a metric is hidden, re-run with `--all-metrics` to include unsupported and
unknown values. Reference comparisons are under `oviq report`; top-level legacy
comparison commands are not part of the current architecture.

## Documentation reference looks stale

Regenerate generated docs before editing them by hand:

```bash
uv run python scripts/docs/build_cli_reference.py
uv run python scripts/docs/build_schema_docs.py
uv run python scripts/docs/build_api_pages.py
uv run python scripts/docs/build_example_bundles.py
```

Generated pages should reflect the CLI, schemas, examples and docstrings in the
same checkout.
