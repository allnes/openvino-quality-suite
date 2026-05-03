# Reports And Gates

Every CLI evaluation command writes a JSON report compatible with `EvaluationReport`.

## Report shape

Top-level fields:

- `schema_version`: report contract version, currently `openvino_llm_quality_v1`.
- `run`: run metadata such as `id`, `suite`, `model`, `reference`, `current`, `device`,
  `precision` and `created_at`.
- `summary`: `overall_status` plus human-readable `main_findings`.
- `inference_equivalence`: logits drift and equivalence metrics.
- `likelihood`: NLL, PPL and likelihood sample metrics.
- `long_context`: long-context and position diagnostics.
- `robustness`: robustness diagnostics.
- `generation`: output-level generation metrics.
- `rag`: retrieval, grounding and citation diagnostics.
- `agent`: tool-use and trace diagnostics.
- `serving`: batch, KV-cache and endpoint diagnostics.
- `performance`: latency, throughput and memory metrics.
- `gates`: evaluated thresholds.
- `metric_references`: reference/oracle metadata for scalar metrics found in the report.
- `reproducibility`: seeds, versions, model hashes and environment notes.
- `analysis`: optional machine-readable findings, normalized metric observations and
  regression lists.
- `artifacts`: optional links to rendered report artifacts.
- `report_metadata`: optional report builder metadata.
- `ui_hints`: optional renderer hints that do not affect metric interpretation.
- `sample_metrics_summary`: optional summary for sample-level detail files.
- `raw_sample_metrics_uri`: optional pointer to detailed sample-level metrics.

The external JSON report envelope is also described by
`src/oviqs/contracts/jsonschema/evaluation_report.schema.json`. The schema fixes the
required envelope fields and allows additional metric-section properties so compatible
consumers can accept newly registered scalar metrics. Reporting metadata fields such as
`analysis.metrics`, `analysis.findings`, `artifacts`, `report_metadata` and
`sample_metrics_summary` are explicitly typed so contract validation catches malformed
reporting output without blocking additive metric evolution.

All report reads pass through the schema normalization layer before comparison, validation
or rendering. Persisted reports must declare the current `openvino_llm_quality_v1`
`schema_version`; missing, older or future versions fail fast instead of being silently
rewritten.

## Status values

Valid status values are `pass`, `warning`, `fail` and `unknown`.

`compare` currently produces `pass`, `warning` or `unknown` from gate evaluation.
Callers can promote warnings to failures in CI when a threshold is release-blocking.

`run-gpu-suite` also writes `status` inside each metric section. A section can be `fail`
when model compilation or inference failed, while other sections still contain their own
results. Treat the top-level `summary.overall_status` as the scorecard status and inspect
section errors before comparing thresholds.

## Gate config

Gate files are YAML objects grouped by report section:

```yaml
likelihood:
  nll_delta_vs_ref_max: 0.10
inference_equivalence:
  mean_logit_cosine_min: 0.995
  abs_mean_entropy_drift_max: 0.15
```

Supported suffixes:

- `_max`: value must be less than or equal to the threshold.
- `_min`: value must be greater than or equal to the threshold.
- `abs_..._max`: absolute value must be less than or equal to the threshold.

When a metric is missing, the check status is `unknown`. When a metric value exists but
has no registered reference/oracle, the check status is also `unknown`; gates are not
allowed to pass unreferenced metrics.

## Metric References

Reports should include `metric_references`. This section records the reference framework,
formula or deterministic oracle used to interpret each metric and its degradation rule.
Metrics without a registered reference are not ready for gates.

Use the CLI to inspect the catalog:

```bash
oviq list-metric-references
oviq list-metric-references --family likelihood --json
```

Nested metrics are referenced by path when needed, for example
`serving.batch_invariance.mean_kl`. A section-level reference such as
`batch_invariance` can cover nested scalar values when the registered oracle applies to
the whole nested metric group.

## Reference Comparisons

Use `oviq report reference-comparison` to compare one or more reports against the same
reference catalog:

```bash
oviq report reference-comparison \
  --report baseline=reports/baseline.json \
  --report current=reports/current.json \
  --format html-dashboard \
  --out reports/reference_comparison.html
```

Supported formats are `markdown`, `markdown-transposed`, `html-dashboard`, `csv` and
`json`. CSV output uses `pandas` when available and falls back to the Python CSV library.
The command reports each metric value, metric status, reference, degradation rule and report
path so target-model results can be compared without losing oracle context.

## Report Bundles

Use the grouped reporting namespace when a run needs CI-friendly and developer-friendly
artifacts:

```bash
oviq report build \
  --report reports/current.json \
  --baseline reports/baseline.json \
  --out reports/run-001 \
  --format all
```

The bundle layout is stable:

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

`analysis.json` contains `MetricObservation` rows, `AnalysisFinding` entries, regression
lists, improvement lists and optional `trend_points` from the configured trend store.
Renderers consume those normalized rows through `ReportViewModel` instead of walking raw
nested metric dictionaries, so adding a metric does not require changing Markdown or HTML
templates.

When sections include per-sample `samples`, the bundle writes them to `sample_metrics.jsonl`
and records numeric outliers in `analysis.sample_outliers`. `metrics.csv` is written through
`pandas` when available, with a stdlib CSV fallback for minimal environments. Markdown and
HTML reports show outliers in the developer view. The HTML dashboard is self-contained and
renders its status chart from embedded JSON data without external network calls.
Sample-metric rows are validated against `sample_metric.schema.json`; validation errors are
recorded in bundle metadata.

`ReportComparisonService` can also consume a `TrendStorePort`. The bundled local adapter
stores report history as JSONL and uses the latest historical report as a baseline when an
explicit baseline is not provided. The `report build`, `report analyze` and
`report metrics-table` commands expose this through `--trend-history`.

Bundle, render, analysis and reference-comparison file writes are routed through artifact
ports. `report build` writes bundles through `ReportBundleWriterPort` after
`ReportGenerationService` enriches the canonical report with analysis metadata.
Reference-comparison output is produced through `ReferenceComparisonRendererPort` and then
persisted through the artifact port. Sample-level detail rows are routed through
`SampleMetricsStorePort`, while scalar metrics are routed through `MetricTableWriterPort`.
Gates are evaluated through `GateEvaluatorPort` so thresholds remain injectable at the
application boundary.

Useful focused commands:

```bash
oviq report analyze --report reports/current.json --gates configs/gates/default_gates.yaml --out reports/analysis.json
oviq report analyze --report reports/current.json --trend-history reports/history.jsonl --out reports/analysis.json
oviq report metrics-table --report reports/current.json --out reports/current.metrics.csv
oviq report render --bundle reports/run-001 --format html-dashboard --out reports/run-001/dashboard.html
oviq report validate --report reports/current.json
```

`report validate` validates the report against the project JSON Schema contract with
`jsonschema` when that dependency is installed, otherwise it falls back to the built-in
minimal validator. It also checks normalized metric observations when `analysis.metrics`
is present.

## Developer Guide

Add a metric:

1. Emit it as a scalar under the appropriate `EvaluationReport` diagnostic section.
2. Register its reference/oracle in the metric reference catalog before using it in gates.
3. Add focused tests for normalization, gates and reference comparison.

Add a reference or oracle:

1. Extend `oviqs.domain.references` with the reference id, family, metric names and
   degradation rule.
2. Keep the rule deterministic enough to decide pass, warning, fail or unknown.
3. Document source framework behavior in `docs/metrics.md` or a metric detail page.

Add an analysis rule:

1. Implement `AnalysisRulePort` in `oviqs.adapters.analysis`.
2. Consume normalized `MetricObservation` rows, not raw runner output.
3. Register the rule through the `oviqs.analysis_rules` entry-point group so
   CLI/reporting uses it through ports.

Add a renderer:

1. Implement `ReportRendererPort` in `oviqs.adapters.reporting`.
2. Consume `ReportViewModel.analysis` view data only; raw reports must be normalized before
   rendering.
3. Add a reporter entry point under `oviqs.reporters` and snapshot tests for stable output.

Extend the schema:

1. Prefer optional additive fields in `evaluation_report.schema.json`.
2. Add a separate schema for new artifact metadata when the object has an independent
   lifecycle.
3. Keep missing or unsupported metrics as `unknown`, never as pass or zero.
