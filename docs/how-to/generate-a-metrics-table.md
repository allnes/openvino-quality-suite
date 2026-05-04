# Generate a metrics table

Create a CSV table from scalar report metrics when reviewers or CI jobs need a
compact artifact instead of the full report bundle:

```bash
oviq report metrics-table --report reports/current.json --out reports/current.metrics.csv
```

The table uses normalized metric paths such as `likelihood.perplexity`.

Use this output when a CI system needs a compact artifact for spreadsheet
review, trend ingestion or quick diffs:

```bash
column -s, -t reports/current.metrics.csv | sed -n '1,20p'
```

The CSV contains:

- `path`: full metric path in the report.
- `section`: top-level diagnostic section.
- `name`: scalar metric name.
- `value`: scalar value or an empty value for `null`.
- `status`: section status inherited while walking the report.

## Expected shape

```csv
path,section,name,value,status
likelihood.perplexity,likelihood,perplexity,2.944679551,warning
likelihood.generation.status,likelihood,generation.status,unknown,unknown
```

The table is intentionally flat. Nested report sections are easier to inspect in
`report.json`; `metrics.csv` is for quick filtering, joins and trend ingestion.

## Compare two reports

Use ordinary shell tools for a first-pass diff:

```bash
oviq report metrics-table --report reports/baseline.json --out reports/baseline.metrics.csv
oviq report metrics-table --report reports/current.json --out reports/current.metrics.csv

diff -u reports/baseline.metrics.csv reports/current.metrics.csv
```

For a report-aware comparison, use:

```bash
oviq report reference-comparison \
  --report baseline=reports/baseline.json \
  --report current=reports/current.json \
  --out reports/reference-comparison.md
```

## Ingest trends

Store `metrics.csv` with the same artifact key as `report.json`. A trend job
should join rows by `path`, `suite`, `model`, `device`, `precision` and dataset
slice. Do not compare values across different metric definitions or dataset
slices unless the report explicitly records that equivalence.

## Troubleshooting

If a metric appears in `report.json` but not in the CSV, the normalization layer
does not recognize that scalar path yet. Add a focused test before changing the
renderer. If a metric is missing from both files, the evaluation service or
backend adapter did not produce it.

The table is derived from the report. Do not edit it by hand; regenerate it from
the source report when values change.
