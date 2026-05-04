# Build a report bundle

Use `oviq report build` when a report needs to be reviewed by a human, uploaded
from CI, or consumed by another tool. The bundle keeps the canonical JSON and
rendered views together.

## Inputs

At minimum, provide a report:

```bash
oviq report build \
  --report /tmp/likelihood.json \
  --out /tmp/likelihood-report \
  --format all
```

To include baseline and gate analysis, add:

```bash
oviq report build \
  --report reports/current.json \
  --baseline reports/baseline.json \
  --gates configs/gates/default_gates.yaml \
  --out reports/current-bundle \
  --format all
```

## Outputs

The bundle layout is stable:

```text
report-bundle/
  report.json
  analysis.json
  metrics.csv
  index.md
  dashboard.html
  assets/
  metadata.json
```

Consumers should read `report.json` for the canonical contract and
`analysis.json` for normalized observations and findings. Rendered files are
views over those artifacts.

## Dashboard preview

![OVIQS dashboard mockup](../assets/mockups/dashboard-mockup.svg)

The HTML dashboard is self-contained so it can be opened from a CI artifact
without network access.

## Validation

Run schema validation before publishing a bundle:

```bash
oviq report validate --report /tmp/likelihood-report/report.json
```

Schema validation checks the public envelope. It does not require every optional
metric to exist.
