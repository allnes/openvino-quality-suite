# Testing and quality

Run the full gate locally:

```bash
uv run pre-commit run --all-files --show-diff-on-failure
```

Documentation changes must also run:

```bash
uv run python scripts/docs/build_cli_reference.py
uv run python scripts/docs/build_schema_docs.py
uv run python scripts/docs/build_api_pages.py
uv run python scripts/docs/build_example_bundles.py
uv run mkdocs build --strict
```

## What the gate protects

The full gate checks:

- generated CLI, API, schema and example docs;
- strict MkDocs build;
- unit and integration tests;
- coverage threshold;
- type checking with mypy and pyright;
- dependency checks, audit and static security checks;
- maintainability gates;
- package build, twine check and wheel smoke.

## Docs-specific expectations

When a change affects CLI help, JSON schemas, public examples or docstrings,
regenerate the relevant docs artifacts. Generated pages should not be edited by
hand.

When a change affects report behavior, update at least one of:

- reporting reference;
- bundle example;
- metric catalogue;
- developer playbook;
- release notes or changelog.

## Local cleanup

MkDocs and Python checks can leave generated directories. Remove local build
artifacts before handing off work:

```bash
rm -rf site
find src tests scripts examples -type d -name __pycache__ -prune -exec rm -rf {} +
```
