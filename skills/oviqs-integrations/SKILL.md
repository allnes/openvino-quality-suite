---
name: oviqs-integrations
description: Use when adding, testing, or running optional OVIQS integration adapters for LightEval, lm-evaluation-harness, OpenCompass, Ragas, DeepEval, Phoenix, Opik, OpenEvals, agentevals, promptfoo, LongBench, HELMET, or RULER.
---

# OVIQS Integrations

Use this skill for optional external evaluator and dataset adapter work.

## References

Read only what is needed:

- `docs/developers/plugin-entry-points.md` for adapter boundaries and plugin entry points.
- `docs/explanation/evaluation-surfaces.md` for evidence requirements and unknown semantics.
- `docs/reference/reporting/reporting-spec.md` for report contract constraints.
- `requirements/integrations.txt` for shared evaluator dependencies.
- `requirements/lighteval.txt` for the separate LightEval environment.
- `src/oviqs/adapters/integrations/base.py` for shared result normalization.
- `src/oviqs/ports/*.py` for public adapter protocols.
- `src/oviqs/adapters/plugins/entrypoints.py` for Python entry-point plugin discovery.
- `tests/unit/test_external_integrations.py` for adapter contract examples.

## Workflow

1. Keep integrations lazy; importing `oviqs` or core metrics must not import optional
   evaluator packages.
2. Return `IntegrationResult` from adapter entry points.
3. Normalize external outputs with `normalize_external_output`.
4. Prefer Python APIs or explicit report importers. Do not launch external CLIs implicitly.
5. Use dataset converters to normalize benchmark rows into `EvalSample`.
6. Keep proxy credentials, evaluator service credentials and generated result files out of git.
7. When exposing an adapter as a package plugin, use the relevant entry-point group
   (`oviqs.runners`, `oviqs.datasets` or `oviqs.reporters`) and keep import-time optional
   dependencies lazy.
8. Keep `requirements/integrations.txt`, `requirements/lighteval.txt`, `pyproject.toml`
   extras and `uv.lock` aligned when dependency pins change.
9. Update MkDocs developer/reference pages when integration behavior becomes public.
10. Run targeted adapter tests while iterating, then the unified `uv run pre-commit
   run --all-files --show-diff-on-failure` gate for release-facing dependency or
   import-boundary changes.

## Install Notes

```bash
uv sync --extra dev
uv run python -m pip install -r requirements/integrations.txt
python -m venv .venv-lighteval
.venv-lighteval/bin/python -m pip install -r requirements/lighteval.txt
```

LightEval uses a separate venv when Phoenix/Opik are installed because their dependency
stacks can require incompatible transitive package ranges.
