# Configure gates

Gate YAML files define threshold checks over scalar metric paths. A gate that
depends on missing or unreferenced evidence should evaluate to `unknown`, not
`pass`.

```bash
oviq report analyze \
  --report reports/current.json \
  --baseline reports/baseline.json \
  --gates configs/gates/default_gates.yaml \
  --out reports/analysis.json
```

## Gate policy

Use gates for metrics with clear reference or oracle metadata. A gate should not
turn missing evidence into success. If the metric is unavailable, the result
should remain `unknown` so reviewers can see that the check was not proven.

Keep gate files small and explicit:

```yaml
likelihood.perplexity:
  max: 30.0
serving.batch_mean_kl:
  max: 0.05
```

Run gate analysis separately when you only need machine-readable output:

```bash
oviq report analyze \
  --report reports/current.json \
  --gates configs/gates/default_gates.yaml \
  --out reports/gated-analysis.json
```

For review, prefer a full bundle so the gate results, findings, metrics and
rendered dashboard stay together.
