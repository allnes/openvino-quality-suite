# Plugin entry points

Package metadata exposes OVIQS extension groups:

- `oviqs.runners`
- `oviqs.datasets`
- `oviqs.reporters`

Plugins should lazy-load optional dependencies and raise clear optional
dependency errors only when selected.

## Current groups

| Group | Purpose |
|---|---|
| `oviqs.runners` | Logits or generation backends. |
| `oviqs.datasets` | Dataset readers and row adapters. |
| `oviqs.reporters` | Report readers, writers and renderers. |

## Rules for plugin authors

- Keep optional imports inside the selected adapter path.
- Return clear errors that name the missing extra or package.
- Keep metric math outside the plugin when it can live in `oviqs.domain`.
- Preserve the `EvaluationReport` schema and bundle layout.
- Add documentation and examples for new public behavior.

## Discovery

Entry points should be treated as public extension hooks. Internal Python module
paths are still pre-release and may change before the first stable release.
