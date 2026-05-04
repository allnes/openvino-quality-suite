# References and oracles

Metrics intended for gating should have reference or oracle metadata. Reports
include a `metric_references` manifest so reviewers can see why a metric is
actionable.

Use:

```bash
oviq list-metric-references --json
```

Filter by family during review:

```bash
oviq list-metric-references --family rag --json
```

## Reference metadata answers

Reference metadata should answer:

- What does the metric measure?
- Which metric family owns it?
- What evidence is required?
- Which thresholds or degradation rules are meaningful?
- What should happen when evidence is missing?

## Gate discipline

Do not gate unreferenced metrics by default. If a gate depends on a metric that
has no reference or oracle, the result should remain `unknown` until the metric
is documented and reviewed.

## Review checklist

When adding or changing a metric reference:

1. Update the catalogue.
2. Add or update tests for gate behavior.
3. Regenerate example bundles if rendered output changes.
4. Mention schema or CLI impact in the pull request.
