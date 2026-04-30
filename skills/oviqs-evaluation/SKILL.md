---
name: oviqs-evaluation
description: Use when running OVIQS diagnostics, long-context, serving, RAG, agent, GPU metric suites, GenAI model matrix commands, metric reference checks, JSONL inputs, report gates, or rendered reports for OpenVINO LLM quality work.
---

# OVIQS Evaluation

Use this skill for hands-on evaluation runs in this repository.

## Workflow

1. Read `README.md`, `docs/usage.md`, `docs/data_formats.md` and
   `docs/reports_and_gates.md` only as needed.
2. Use the project virtual environment for Python commands: `.venv/bin/python`,
   `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/pre-commit` or `.venv/bin/oviq`.
3. Choose the backend:
   - `dummy` for deterministic local checks.
   - `hf` for reference logits.
   - `optimum-openvino` or `openvino-runtime` for logits-level OpenVINO checks.
   - Generation or serving adapters only for output-level checks.
4. Use `oviq run-gpu-suite` for GPU scorecards when an exported OpenVINO logits model
   and target GPU environment are available.
5. Use `oviq eval-long-context`, `oviq eval-serving`, `oviq eval-rag` and
   `oviq eval-agent` when validating specific diagnostic surfaces from JSONL inputs.
6. Use `oviq list-genai-models` and `oviq genai-export-plan` with
   `configs/examples/genai_metric_models.yaml` to choose models and export variants.
7. Use `oviq list-metric-references` before trusting or changing gates.
8. Write generated reports outside source paths, preferably `/tmp` or ignored `reports/`.
9. Run `oviq compare` with `configs/gates/default_gates.yaml` when thresholds matter.
10. Render Markdown with `oviq render-report` when a single human-readable report is needed.
11. Use `oviq reference-comparison` when comparing metric values across several reports.

## Guardrails

- Do not infer KL, JS, entropy drift, logit cosine, NLL or PPL from text-only generation.
- Treat logits shape mismatches as real alignment failures.
- Treat GPU compile failures as report findings, not as successful metric values.
- Keep rule-based RAG/agent values separate from external judge-backed values.
- Treat gate checks with missing metric references as `unknown`, not pass.
- Leave judge-backed RAG or agent metrics missing unless the evaluator actually ran.
- Do not commit generated reports, caches, model exports or virtual environments.
- Keep coverage files, lockfile refreshes and CI artifacts separate from evaluation output
  unless the change explicitly updates release tooling.

## Minimal commands

```bash
.venv/bin/oviq eval-likelihood --model dummy --backend dummy --dataset /tmp/likelihood.jsonl --out /tmp/likelihood.json
.venv/bin/oviq eval-serving --model dummy --backend dummy --out /tmp/serving.json
.venv/bin/oviq eval-rag --dataset /tmp/rag.jsonl --answers /tmp/rag_answers.jsonl --out /tmp/rag_report.json
.venv/bin/oviq eval-agent --traces /tmp/agent_traces.jsonl --expected /tmp/agent_expected.jsonl --out /tmp/agent_report.json
.venv/bin/oviq run-gpu-suite --model models/sshleifer--tiny-gpt2-eval_logits --backend openvino-runtime --dataset /tmp/likelihood.jsonl --device GPU --out /tmp/gpu_suite.json
.venv/bin/oviq list-genai-models --tier smoke --metric likelihood
.venv/bin/oviq list-metric-references --family rag --json
.venv/bin/oviq compare --baseline /tmp/likelihood.json --current /tmp/likelihood.json --gates configs/gates/default_gates.yaml --out /tmp/comparison.json
.venv/bin/oviq render-report --report /tmp/likelihood.json --out /tmp/likelihood.md
.venv/bin/oviq reference-comparison --report baseline=/tmp/baseline.json --report current=/tmp/current.json --format markdown-transposed --out /tmp/reference_comparison.md
.venv/bin/pre-commit run --all-files
```
