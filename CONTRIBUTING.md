# Contributing

## Quality Checks

This repository uses `pre-commit` as the single mandatory quality gate.

Install development dependencies:

```bash
uv sync --extra dev
```

Install git hooks:

```bash
uv run pre-commit install
```

Run the full quality suite manually:

```bash
uv run pre-commit run --all-files --show-diff-on-failure
```

The CI workflow runs the same pre-commit suite.
