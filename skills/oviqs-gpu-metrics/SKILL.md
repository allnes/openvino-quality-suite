---
name: oviqs-gpu-metrics
description: Use when preparing or running OVIQS GPU metric verification, standard metric matrix runs, OpenVINO Runtime GPU scorecards, OpenVINO GenAI export plans, canonical OpenVINO/llm target model checks (Qwen3-0.6B, gemma-2-9b-it, Mistral-7B-Instruct-v0.1, gpt-oss-20b, phi-4), clean remote GPU workflows, or extended GPU metric scripts.
---

# OVIQS GPU Metrics

Use this skill for GPU-specific quality verification in this repository.

## References

Read only what is needed:

- `docs/tutorials/run-the-gpu-suite.md` for the public GPU workflow.
- `docs/reference/cli/run-gpu-suite.md` for current CLI flags.
- `docs/reference/metrics/playbook.md` for formulas, dataset requirements and triage actions.
- `docs/reference/metrics/catalogue.md` for GPU-relevant metric paths and unknown semantics.
- `docs/how-to/integrate-in-ci.md` for report validation, bundling and CI artifact handling.
- `configs/examples/genai_metric_models.yaml` for the canonical five OpenVINO/llm models.
- `configs/suites/gpu_metric_smoke.yaml` for the GPU scorecard scope.
- `scripts/remote_gpu_target_models.py` for the canonical target-model GPU runs.
- `scripts/remote_gpu_extended_metrics.py` for WikiText-2, drift, performance,
  RAG and agent extended metrics.
- `scripts/remote_gpu_standard_metric_matrix.py` for broad reference-backed metric
  coverage across likelihood, drift, long-context, generation, serving, RAG and agent.

## Workflow

1. Use `.venv/bin/...` locally and the target GPU-machine venv in remote scripts.
2. Pick a model-preparation mode (the remote scripts take `MODE=openvino|convert`):
   - `MODE=openvino` (default): download the ready-made OpenVINO INT4 artifact from the
     OpenVINO/llm collection (`huggingface-cli download OpenVINO/<model>-int4-ov`). These
     artifacts expose full causal-LM logits for OpenVINO Runtime and a with-past variant for
     OpenVINO GenAI, so no export step is required.
   - `MODE=convert`: direct conversion of the base checkpoint with `optimum-cli export
     openvino` (`text-generation` for the logits model, `text-generation-with-past` for the
     GenAI generation model).
3. Use `oviq genai-export-plan` to generate the convert-mode export commands for a base
   checkpoint when you are not downloading the published artifact.
4. Run `oviq run-gpu-suite` against the prepared model directory.
5. Pass `--genai-model` (same directory in openvino mode, the with-past export in convert mode)
   for the generation layer.
6. Use `Qwen/Qwen3-0.6B` (OpenVINO/Qwen3-0.6B-int4-ov) for the smallest GPU sanity run.
7. Use a documented target GPU device ID when validating larger target model behavior.
8. Use the standard metric matrix script when validating reference coverage across metric
   families.
9. Use `oviq report reference-comparison` to compare standard matrix reports across target
   models.
10. Build review bundles with `oviq report build` when publishing GPU results for humans.
11. Store generated reports under ignored `reports/` or remote workspace report paths.
12. Keep `uv.lock`, requirements files and CI docs aligned when GPU or evaluator
    dependencies change.

## Guardrails

- Do not commit exported models, reports, virtual environments or downloaded caches.
- Do not run target GPU scripts on a local CPU-only development environment.
- Do not replace requested target model families without documenting OpenVINO GenAI support
  status and the exact fallback model or artifact.
- Keep judge-backed RAG and agent metrics `unknown` unless an explicit scorer ran.
- Treat OpenVINO GPU compile errors as failed section results that need investigation.
- Keep local device inventories and raw GPU reports out of commits; publish sanitized
  summaries in docs only when they are reproducible from public configs.

## Commands

```bash
.venv/bin/oviq list-genai-models --config configs/examples/genai_metric_models.yaml --tier target_gpu
huggingface-cli download OpenVINO/Qwen3-0.6B-int4-ov --local-dir models/qwen3-0_6b-int4-ov
.venv/bin/oviq run-gpu-suite --model models/qwen3-0_6b-int4-ov --genai-model models/qwen3-0_6b-int4-ov --backend openvino-runtime --dataset /tmp/likelihood.jsonl --device GPU --window-size 64 --stride 32 --out reports/gpu_metric_suite.json
PYTHONPATH=src .venv/bin/python scripts/remote_gpu_standard_metric_matrix.py --model models/qwen3-0_6b-int4-ov --genai-model models/qwen3-0_6b-int4-ov --dataset-cache data/standard-matrix --out reports/standard_metric_matrix.json
.venv/bin/oviq report build --report reports/standard_metric_matrix.json --out reports/standard_metric_matrix-bundle --format all
.venv/bin/oviq report reference-comparison --report qwen3=reports/standard_metric_matrix.json --format html-dashboard --out reports/standard_metric_matrix.html
```
