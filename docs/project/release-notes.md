# Release notes

OVIQS is pre-release. The project does not preserve backward compatibility for
internal Python imports before the first stable release. Breaking changes are
allowed while the architecture is being finalized, but user-facing report
contracts must be described clearly.

Each release note should include:

- summary
- breaking changes
- schema and CLI impact
- migration notes
- contributors

## Template

```markdown
# OVIQS <version>

## Summary

Short release narrative.

## Compatibility

- Report schema impact:
- Bundle layout impact:
- CLI impact:
- Python import path impact:

## Added

## Changed

## Fixed

## Migration notes

## Contributors
```

## Pre-release rule

Before the first stable release, internal Python import paths may change without
a compatibility shim. Release notes should still call out changes to report
schemas, bundle layout and CLI commands because those are user-facing contracts.

The project is new, so legacy compatibility shims should not be added only to
preserve old internal module paths or old top-level report commands. Prefer the
current architecture and document the new path.

## First stable release checklist

Before tagging the first stable release:

1. Confirm every CLI command in `docs/reference/cli/` is generated from the
   current Typer app.
2. Confirm every schema page in `docs/reference/schemas/` is generated from
   `src/oviqs/contracts/jsonschema/`.
3. Validate all generated example bundles.
4. Build the documentation with `mkdocs build --strict`.
5. Build the package artifacts and check that local operator files are not
   included in the distribution.
6. Record any public CLI, schema or bundle changes in the release note.

## Documentation rule

Each release should point to the matching documentation version. Read the Docs
will provide the stable/latest model after the first tagged release.

Use `latest` for the main branch documentation and `stable` for the most recent
stable tag after releases are enabled. Pre-release docs may change together with
the code in `main`.
