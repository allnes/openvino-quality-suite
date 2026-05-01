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
- `raw_sample_metrics_uri`: optional pointer to detailed sample-level metrics.

The external JSON report envelope is also described by
`src/oviqs/contracts/jsonschema/evaluation_report.schema.json`. The schema fixes the
required envelope fields and allows additional metric-section properties so compatible
consumers can accept newly registered scalar metrics.

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

Use `oviq reference-comparison` to compare one or more reports against the same reference
catalog:

```bash
oviq reference-comparison \
  --report baseline=reports/baseline.json \
  --report current=reports/current.json \
  --format html-dashboard \
  --out reports/reference_comparison.html
```

Supported formats are `markdown`, `markdown-transposed`, `html-dashboard`, `csv` and
`json`. The command reports each metric value, metric status, reference, degradation rule
and report path so target-model results can be compared without losing oracle context.
