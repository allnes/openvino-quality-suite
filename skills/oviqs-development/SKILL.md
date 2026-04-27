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
5. Keep `src/oviqs/models` tracked; `/models/` is the ignored generated artifact root.
6. Keep optional integrations lazy. Shared behavior belongs in `oviqs.integrations.base`;
   adapter modules should return `IntegrationResult` and avoid implicit subprocess calls.
7. Register new gateable scalar metrics in `oviqs.references`; reports and gates rely on
   reference/oracle metadata to decide whether a metric is quality-gate ready.
8. Add focused tests for metric math, schemas, gates, metric references, model-matrix
   behavior, adapters and CLI behavior affected by the change.
9. Update `docs/metric_playbook.md` and the matching `docs/metric_details/*.md` file when
   adding or renaming a reported metric.
10. Run checks from the virtual environment.

## Validation

Prefer the narrowest checks that cover the change:

```bash
.venv/bin/pytest tests/unit
.venv/bin/ruff check .
```

Run integration or OpenVINO-specific tests only when the required models, devices and
extras are available.

## Design guardrails

- Metrics should operate on normalized arrays, reports or traces, not on backend-specific objects.
- Runner adapters should translate framework outputs into OVIQS core contracts.
- RAG, agent, serving and long-context CLI commands should compute rule-based metrics when
  inputs are available and keep judge/stateful-only metrics explicit when unavailable.
- Gate evaluation should leave missing or unreferenced metrics as `unknown`.
- OpenVINO Runtime GPU compile failures should be captured as report failures when the
  command is a scorecard-style workflow.
- Missing optional evaluators should produce omitted or `unknown` metrics, not fake scores.
- Report JSON must remain stable enough for CI gates and downstream rendering.
