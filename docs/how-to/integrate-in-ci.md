# Integrate in CI

Recommended CI sequence:

1. Run the evaluation command.
2. Validate the report with `oviq report validate`.
3. Build a report bundle with `oviq report build`.
4. Upload the bundle as an artifact.
5. Run `mkdocs build --strict` for documentation changes.

## Minimal shell sequence

```bash
oviq eval-likelihood \
  --model dummy \
  --backend dummy \
  --dataset tests/fixtures/likelihood.jsonl \
  --out reports/current.json

oviq report validate --report reports/current.json

oviq report build \
  --report reports/current.json \
  --out reports/current-bundle \
  --format all
```

## Documentation changes

Documentation pull requests should regenerate reference artifacts and build the
site strictly:

```bash
uv run python scripts/docs/build_cli_reference.py
uv run python scripts/docs/build_schema_docs.py
uv run python scripts/docs/build_api_pages.py
uv run python scripts/docs/build_example_bundles.py
uv run mkdocs build --strict
```

The repository pre-commit gate runs these checks, along with tests, type checks,
security checks and package smoke tests:

```bash
uv run pre-commit run --all-files --show-diff-on-failure
```

Upload `reports/current-bundle/` as a CI artifact. Reviewers should not need to
re-run the evaluation to inspect the Markdown or HTML output.
