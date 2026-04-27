---
name: oviqs-gpu-metrics
description: Use when preparing or running OVIQS GPU metric verification, standard metric matrix runs, OpenVINO Runtime GPU scorecards, OpenVINO GenAI export plans, validated GPT-2/Gemma/Qwen target checks, clean remote GPU workflows, or extended GPU metric scripts.
---

# OVIQS GPU Metrics

Use this skill for GPU-specific quality verification in this repository.

## References

Read only what is needed:

- `docs/gpu_requirements.md` for target GPU environment dependencies and status.
- `docs/remote_gpu_from_scratch.md` for clean remote workspace setup.
- `docs/genai_model_matrix.md` for model tiers and export variants.
- `configs/examples/genai_metric_models.yaml` for machine-readable model choices.
- `configs/suites/gpu_metric_smoke.yaml` for the GPU scorecard scope.
- `scripts/remote_gpu_validated_gpt2.sh` for a validated GPT-2 base GPU run.
- `scripts/remote_gpu_extended_gpt2_metrics.py` for WikiText-2, drift, performance,
  RAG and agent extended metrics.
- `scripts/remote_gpu_standard_metric_matrix.py` for broad reference-backed metric
  coverage across likelihood, drift, long-context, generation, serving, RAG and agent.

## Workflow

1. Use `.venv/bin/...` locally and the target GPU-machine venv in remote scripts.
2. Keep evaluation and generation artifacts separate:
   - `text-generation` export for OpenVINO Runtime logits metrics.
   - `text-generation-with-past` export for OpenVINO GenAI generation checks.
3. Generate export commands with `oviq genai-export-plan` before converting models.
4. Run `oviq run-gpu-suite` against an exported logits model directory.
5. Pass `--genai-model` only when a compatible GenAI export exists.
6. Use `openai-community/gpt2` for the currently validated non-toy GPU sanity run.
7. Use a documented target GPU device ID when validating larger target model behavior.
8. Use the standard metric matrix script when validating reference coverage across metric
   families.
9. Use `oviq reference-comparison` to compare standard matrix reports across target models.
10. Store generated reports under ignored `reports/` or remote workspace report paths.

## Guardrails

- Do not commit exported models, reports, virtual environments or downloaded caches.
- Do not run target GPU scripts on a local CPU-only development environment.
- Do not replace requested target model families without documenting OpenVINO GenAI support
  status and the exact fallback model or artifact.
- Keep judge-backed RAG and agent metrics `unknown` unless an explicit scorer ran.
- Treat OpenVINO GPU compile errors as failed section results that need investigation.

## Commands

```bash
.venv/bin/oviq list-genai-models --config configs/examples/genai_metric_models.yaml --tier smoke
.venv/bin/oviq list-genai-models --config configs/examples/genai_metric_models.yaml --tier validated_gpu
.venv/bin/oviq genai-export-plan --model openai-community/gpt2 --variant eval_logits --variant genai_generation
.venv/bin/oviq run-gpu-suite --model models/sshleifer--tiny-gpt2-eval_logits --backend openvino-runtime --dataset /tmp/likelihood.jsonl --device GPU --out reports/gpu_metric_suite.json
PYTHONPATH=src .venv/bin/python scripts/remote_gpu_standard_metric_matrix.py --model models/openai-community--gpt2-eval-fp16 --genai-model models/openai-community--gpt2-genai-fp16 --dataset-cache data/standard-matrix --out reports/standard_metric_matrix.json
.venv/bin/oviq reference-comparison --report gpt2=reports/standard_metric_matrix.json --format html-dashboard --out reports/standard_metric_matrix.html
```
