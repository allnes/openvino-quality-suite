# CI/CD

The default public workflow lives in `.github/workflows/quality.yml`. It uses
`uv run pre-commit run --all-files --show-diff-on-failure` as the single mandatory quality
gate. The same command is the expected local pre-push check.

Run the full local gate before pushing:

```bash
uv run pre-commit run --all-files --show-diff-on-failure
```

Run OpenVINO tests only in environments with exported models and configured devices.

## Workflow Jobs

- Full pre-commit quality gate: installs with `uv`, syncs the `dev` extra and runs every
  hook from `.pre-commit-config.yaml`.
- Python compatibility matrix: runs `pytest -q` on Python 3.10, 3.11 and 3.12 after the
  full quality gate passes.

## Local Tooling

Development tooling is declared in the `dev` optional dependency group and resolved through
`uv.lock`. Pre-commit is organized into sections:

- file hygiene and syntax/config validation;
- debug statement and private-key checks;
- `uv` lock validation;
- Ruff lint and format;
- ShellCheck and markdownlint-cli2;
- project-local checks through `uv run`: compile, `mypy`, `pyright`, `pytest`, coverage,
  `deptry`, `pip check`, `pip-audit`, Bandit, codespell, docstring checks, vulture, radon,
  xenon, package build, `twine check`, wheel install smoke, `import oviqs` and `oviq --help`.

The package build smoke test is implemented in `scripts/quality/package_smoke.py` so the
pre-commit hook stays deterministic and short.

## Artifacts

Do not commit generated reports, model exports, local datasets, logs, coverage files,
build outputs or caches; those paths are ignored for local and CI output.
