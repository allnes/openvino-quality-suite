# Run a suite

Suite YAML files live under `configs/suites/`. Use them when the same model,
dataset and gate profile should be exercised repeatedly in CI or release checks.

## Choose the suite

Start with a checked-in suite profile:

```bash
find configs/suites -maxdepth 1 -name '*.yaml' -print
```

The suite should describe a stable evaluation intent: model inputs, dataset
slice, backend/device choices, and any gate profile used for publication. Keep
ad-hoc local experiments in individual `eval-*` commands until the workflow is
stable enough to commit.

Run one with:

```bash
oviq run-suite \
  --config configs/suites/openvino_llm_quality_v1.yaml \
  --out reports/suite.json
```

Validate and package the result:

```bash
oviq report validate --report reports/suite.json

oviq report build \
  --report reports/suite.json \
  --out reports/suite-bundle \
  --format all
```

Keep the suite config under version control. Treat the generated report and
bundle as artifacts, not as hand-edited source files.

Use `run-suite` for broad smoke and regression checks. Use the individual
`eval-*` commands when you are developing a specific metric family and want a
tighter feedback loop.

## Minimal suite review

Before using a suite in CI, check:

1. The suite name maps to the intended report `run.suite`.
2. Dataset paths are relative to the repository or documented artifact mount.
3. Backend names match installed extras on the runner.
4. Device-specific suites record the intended `device` and `precision`.
5. Gate files only reference metrics with reference or oracle metadata.

## CI pattern

```bash
oviq run-suite \
  --config configs/suites/openvino_llm_quality_v1.yaml \
  --out reports/suite.json

oviq report build \
  --report reports/suite.json \
  --gates configs/gates/default_gates.yaml \
  --out reports/suite-bundle \
  --format all
```

Upload both `reports/suite.json` and `reports/suite-bundle/`. The JSON report is
the canonical machine artifact; the bundle is the review artifact.

## When not to use a suite

Do not use `run-suite` while debugging a single metric calculation. Run the
specific command, regenerate its metrics table, and only promote the case into a
suite after the expected report paths and missing-evidence behavior are clear.
