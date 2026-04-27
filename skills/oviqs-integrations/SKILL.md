---
name: oviqs-integrations
description: Use when adding, testing, or running optional OVIQS integration adapters for LightEval, lm-evaluation-harness, OpenCompass, Ragas, DeepEval, Phoenix, Opik, OpenEvals, agentevals, promptfoo, LongBench, HELMET, or RULER.
---

# OVIQS Integrations

Use this skill for optional external evaluator and dataset adapter work.

## References

Read only what is needed:

- `docs/integrations.md` for adapter boundaries and install guidance.
- `requirements/integrations.txt` for shared evaluator dependencies.
- `requirements/lighteval.txt` for the separate LightEval environment.
- `src/oviqs/integrations/base.py` for shared result normalization.
- `tests/unit/test_external_integrations.py` for adapter contract examples.

## Workflow

1. Keep integrations lazy; importing `oviqs` or core metrics must not import optional
   evaluator packages.
2. Return `IntegrationResult` from adapter entry points.
3. Normalize external outputs with `normalize_external_output`.
4. Prefer Python APIs or explicit report importers. Do not launch external CLIs implicitly.
5. Use dataset converters to normalize benchmark rows into `EvalSample`.
6. Keep proxy credentials, evaluator service credentials and generated result files out of git.

## Install Notes

```bash
.venv/bin/python -m pip install -r requirements/integrations.txt
python -m venv .venv-lighteval
.venv-lighteval/bin/python -m pip install -r requirements/lighteval.txt
```

LightEval uses a separate venv when Phoenix/Opik are installed because their dependency
stacks can require incompatible transitive package ranges.
