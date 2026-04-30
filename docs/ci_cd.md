# CI/CD

The default public workflow lives in `.github/workflows/quality.yml`. It is designed to
keep core checks lightweight while still covering formatting, typing, dependency health,
security, packaging and tests.

Run the minimum local checks before pushing:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest
git diff --check
```

Run OpenVINO tests only in environments with exported models and configured devices.

## Workflow Jobs

- Static analysis: `ruff check`, `ruff format --check`, pre-commit hooks, syntax compile,
  `mypy` and `pyright`.
- Dependency health: `uv lock --locked`, `deptry`, `pip check` and `pip-audit`.
- Maintainability: `codespell`, `interrogate`, `pydoclint`, `vulture`, `radon` and
  `xenon`.
- Tests: `pytest -q` across Python 3.10, 3.11 and 3.12.
- Coverage: `pytest --cov=oviqs` with the project coverage gate.
- Security: Gitleaks over the checkout plus Bandit over `src`.
- Package build: source/wheel build, `twine check` and installed-wheel smoke test.

## Local Tooling

Development tooling is declared in the `dev` optional dependency group. Pre-commit uses:

- general file hygiene hooks from `pre-commit-hooks`;
- Ruff lint and format hooks;
- ShellCheck for shell scripts;
- markdownlint-cli2 with `.markdownlint-cli2.yaml`.

Use `uv.lock` as the checked dependency lock for CI dependency-health validation.

## Artifacts

Persist generated JSON reports, rendered Markdown reports, comparison outputs and coverage
files as CI artifacts when needed. Do not commit generated reports, model exports, local
datasets, logs or caches; those paths are ignored for local and CI output.
