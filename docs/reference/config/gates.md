# Gates

Gates map metric paths to threshold rules. A missing or unsupported metric should
produce `unknown`.

Default gates live in `configs/gates/default_gates.yaml`.

## Example

```yaml
likelihood.perplexity:
  max: 30.0
serving.batch_mean_kl:
  max: 0.05
```

Run the gate file through analysis:

```bash
oviq report analyze \
  --report reports/current.json \
  --gates configs/gates/default_gates.yaml \
  --out reports/gated-analysis.json
```

## Path rules

Use normalized metric paths from `metrics.csv` or `analysis.json`. If a path is
wrong, the gate should not silently pass.

## Review rules

- Gate only metrics with clear reference or oracle metadata.
- Keep local thresholds visible in version-controlled YAML.
- Prefer a small number of high-signal gates.
- Treat `unknown` as actionable review evidence.
