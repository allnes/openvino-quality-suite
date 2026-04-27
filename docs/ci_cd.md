# CI/CD

Run core tests on every change:

```bash
.venv/bin/pytest
.venv/bin/ruff check .
```

Run OpenVINO tests only in environments with exported models and configured devices.

## Suggested jobs

- Unit checks: install `.[dev]`, run `pytest tests/unit` and `ruff check .`.
- Integration smoke: run dummy CLI flows and report rendering.
- OpenVINO smoke: run tests marked `openvino` only on workers with exported models,
  configured devices and the `openvino` extra installed.
- GPU metric smoke: run `run-gpu-suite` only on workers with OpenVINO GPU devices and
  exported model artifacts. Persist the JSON scorecard as a CI artifact.
- Optional integrations: split LightEval, lm-eval, Ragas, Phoenix, Opik and other
  framework checks into opt-in jobs so core diagnostics stay lightweight.

## Artifacts

Persist JSON reports, rendered Markdown reports and comparison outputs from CI.
Avoid committing generated reports under `reports/`; that path is ignored for local
and CI output.
