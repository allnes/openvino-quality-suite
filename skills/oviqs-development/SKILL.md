---
name: oviqs-development
description: Use when implementing or updating OVIQS code, GPU metrics, GenAI model matrix support, metric references, optional integration adapters, runners, report schemas, CLI commands, tests, or documentation in the OpenVINO Inference Quality Suite repository.
---

# OVIQS Development

Use this skill for code and documentation changes in this repository.

## Workflow

1. Inspect the existing module before editing; prefer local patterns over new abstractions.
2. Keep logits-level diagnostics separate from generation, serving, RAG and agent layers.
3. Preserve optional dependency boundaries. Core imports must not require OpenVINO,
   PyTorch, evaluation frameworks or observability packages unless that adapter is used.
4. Update docs and examples when public CLI flags, runner names, data formats, report
   sections or gate semantics change.
5. Keep `src/oviqs/domain/models` tracked; `/models/` is the ignored generated artifact root.
6. Keep the layered boundaries intact: interfaces call application services, application
   code depends on domain and ports, and concrete runner/dataset/report implementations
   stay in adapters.
7. Keep optional integrations lazy. Shared evaluator behavior belongs in
   `oviqs.adapters.integrations.base`; infrastructure adapters live under
   `oviqs.adapters` and should satisfy ports without importing optional extras until used.
8. Register new gateable scalar metrics in `oviqs.domain.references`; reports and gates
   rely on reference/oracle metadata to decide whether a metric is quality-gate ready.
9. Add focused tests for metric math, schemas, gates, metric references, model-matrix
   behavior, adapters, architecture boundaries and CLI behavior affected by the change.
10. Update `docs/metric_playbook.md` and the matching `docs/metric_details/*.md` file when
   adding or renaming a reported metric.
11. Keep architecture, usage, contract and skills docs current when package layout, entry
    points, profiles, ports or public interfaces change.
12. Keep reporting changes split across domain reporting models, application reporting
    services, ports and adapters; CLI commands should call `ReportWorkflowService`.
13. Keep CI and packaging metadata current when dependencies or validation tooling change.
14. Run the unified quality gate with `uv run pre-commit run --all-files`.
15. When editing CI, keep diagnostic uploads limited to generated public-safe artifacts such
    as pre-commit logs, coverage XML and package distributions; keep local paths and
    operator notes out of committed docs and skills, and use short retention for
    diagnostic-only artifacts.

## Validation

Prefer the narrowest checks while iterating:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/unit
```

Run integration or OpenVINO-specific tests only when the required models, devices and
extras are available.

For release-facing changes, run the same gate CI uses:

```bash
uv run pre-commit run --all-files --show-diff-on-failure
```

The package build, wheel install smoke test, `import oviqs` and `oviq --help` checks live in
`scripts/quality/package_smoke.py`.

## Design guardrails

- Metrics should operate on normalized arrays, reports or traces, not on backend-specific objects.
- Runner adapters should translate framework outputs into OVIQS domain contracts.
- New delivery surfaces should add a port or interface before coupling application services
  to concrete infrastructure.
- Report renderers should consume `ReportViewModel`; analysis, gates and outlier detection
  belong in `oviqs.application.reporting`, not in renderer templates.
- RAG, agent, serving and long-context CLI commands should compute rule-based metrics when
  inputs are available and keep judge/stateful-only metrics explicit when unavailable.
- Gate evaluation should leave missing or unreferenced metrics as `unknown`.
- OpenVINO Runtime GPU compile failures should be captured as report failures when the
  command is a scorecard-style workflow.
- Missing optional evaluators should produce omitted or `unknown` metrics, not fake scores.
- Report JSON must remain stable enough for CI gates and downstream rendering.
- Generated reports, model exports, datasets, logs, caches, local environment files and
  operator notes must stay out of commits.
- CI artifacts are for workflow diagnosis and release packaging only; never make generated
  artifacts part of the tracked source tree.
- Security suppressions such as `# nosec` should be narrow and explain intentional external
  model or endpoint loading behavior.
