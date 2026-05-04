# Contributing

This page mirrors the root contribution guide and adds docs impact rules.

Every pull request should state:

- docs impact
- schema impact
- CLI surface impact
- whether example bundles were updated
- whether a release note is needed

## Local checks

Run the full quality gate before requesting review:

```bash
uv run pre-commit run --all-files --show-diff-on-failure
```

For documentation-only changes, also verify that generated artifacts are current:

```bash
uv run python scripts/docs/build_cli_reference.py
uv run python scripts/docs/build_schema_docs.py
uv run python scripts/docs/build_api_pages.py
uv run python scripts/docs/build_example_bundles.py
uv run mkdocs build --strict
```

## Docs impact

Update documentation when a change affects:

- CLI commands or options;
- JSON Schema contracts;
- report bundle files;
- metric names, paths or status semantics;
- gate behavior;
- extension points;
- installation or CI behavior.

Generated reference pages should be regenerated, not edited by hand.

## Review expectations

Keep pull requests scoped. If a code change also changes docs architecture,
schemas and CLI behavior, explain the dependency between those changes in the
summary.

Because OVIQS is pre-release, do not preserve old internal module paths through
compatibility shims. Update imports, tests and docs to the target architecture.
