# Developer documentation

Use these pages when changing OVIQS internals.

Architecture boundaries are enforced by tests. New code should preserve the
layered package layout and avoid legacy import shims.

## Common workflows

- [Add a metric](adding-a-metric.md) when new backend-independent quality math
  is needed.
- [Add an analysis rule](adding-an-analysis-rule.md) when existing metrics need
  a reviewer-facing finding.
- [Add a renderer](adding-a-renderer.md) when reports need another output view.
- [Plugin entry points](plugin-entry-points.md) describes externally
  discoverable extension groups.
- [Testing and quality](testing-and-quality.md) lists the local gates expected
  before a pull request.

## Boundary rules

Keep deterministic metric semantics in `oviqs.domain`. Put orchestration in
`oviqs.application`. Depend on protocols from `oviqs.ports` when application
code needs external capabilities. Put concrete integrations, runners and
renderers under `oviqs.adapters`.

The public compatibility surface is the report schema, bundle layout and CLI
behavior. Do not add compatibility import shims for old internal module paths.
