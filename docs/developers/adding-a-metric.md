# Add a metric

Use this workflow for a new scalar diagnostic that should appear in
`EvaluationReport`, reporting bundles or gates.

## Checklist

1. Add backend-independent metric logic in `oviqs.domain.metrics`.
2. Add application wiring that places the scalar in `EvaluationReport`.
3. Register reference or oracle metadata when the metric is gateable.
4. Add tests for math, report output and gates.
5. Update the metric catalogue and examples.

## Domain first

Metric math should accept plain Python data structures or arrays and return a
deterministic value. Avoid importing runners, adapters or optional framework
packages from `oviqs.domain`.

## Report path

Choose a normalized metric path before wiring the report. Use section-qualified
paths such as `likelihood.perplexity` or `serving.batch_mean_kl`. The same path
should appear in:

- the nested `EvaluationReport` section;
- `analysis.json` metric observations;
- `metrics.csv`;
- gate files when the metric is gateable;
- metric catalogue documentation.

## Missing evidence

If a backend cannot provide the evidence required by the metric, emit `unknown`
status or omit the scalar while preserving a finding that explains the missing
evidence. Do not approximate a metric from unrelated evidence.

## Tests

Add focused tests for:

- metric math edge cases;
- evaluation service report output;
- schema-valid report examples;
- gate behavior for pass, warning, fail and unknown where applicable.

## Example skeleton

Keep the pure calculation separate from report wiring:

```python
def normalized_delta(reference: float, current: float) -> float:
    if reference == 0:
        raise ValueError("reference must be non-zero")
    return (current - reference) / abs(reference)
```

Then wire the value into an application service that already owns the
`EvaluationReport` construction. The metric path should be stable before it is
used in gates:

```python
report["drift"]["normalized_delta"] = {
    "value": normalized_delta(reference_score, current_score),
    "status": "warning",
}
```

Finally, add a generated or curated example that includes the path in both the
nested report section and the flattened metrics table. This keeps the public
contract visible in docs and tests.
