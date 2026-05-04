# Use trend history

Trend history is a reporting extension point. Until a dedicated trend store is
configured, store generated reports and bundles as CI artifacts, then compare the
current report against a selected baseline.

Use `--baseline` with `oviq report build` or `oviq report analyze`:

```bash
oviq report analyze \
  --report reports/current.json \
  --baseline artifacts/main/latest/report.json \
  --out reports/trend-analysis.json
```

For reviewable output, build a full bundle:

```bash
oviq report build \
  --report reports/current.json \
  --baseline artifacts/main/latest/report.json \
  --out reports/current-bundle \
  --format all
```

Use comparable baselines:

- same suite name;
- same metric definitions;
- same dataset slice where possible;
- same device and precision when the metric depends on runtime behavior.

If the baseline lacks a metric, keep the delta unknown. Do not backfill missing
values from another metric family.

## Artifact layout

Use a stable artifact layout so CI can find the intended baseline:

```text
artifacts/
  main/
    latest/
      report.json
      metrics.csv
      bundle/
  pull-requests/
    123/
      report.json
      metrics.csv
      bundle/
```

The raw `report.json` is the best input for future comparisons. The full bundle
is the best artifact for human review.

## Select the baseline

Use a release baseline for release gates, a `main/latest` baseline for pull
request checks, and a device-specific baseline for GPU or NPU regressions. Do
not compare a CPU run to a GPU run unless the metric explicitly describes a
device-independent quality surface.

## Debug bad trend output

If every delta is unknown, check that the baseline file exists and validates:

```bash
oviq report validate --report artifacts/main/latest/report.json
oviq report validate --report reports/current.json
```

If only one family is unknown, inspect `metrics.csv` for the exact path spelling
in both reports. Trend comparison is path-based; `likelihood.perplexity` and
`likelihood.ppl` are different public metrics until a documented alias exists.
