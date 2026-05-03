# OVIQS: OpenVINO Inference Quality Suite

OVIQS is an OpenVINO-first diagnostic library for LLM inference quality. It separates
logits-level metrics such as NLL, PPL, KL, JS and entropy drift from generation,
serving, RAG and agent-level evaluation.

Project status: early `0.1.0` implementation scaffold with working core metrics,
runner interfaces, CLI commands, gates and report rendering. OpenVINO-dependent
paths are optional and require exported models plus the matching extras.

## Quick start

```bash
python -m pip install -e ".[dev]"
oviq --help
pytest
```

Minimal likelihood run with the deterministic dummy runner:

```bash
printf '{"id":"s1","task_type":"likelihood","text":"this is a test"}\n' > /tmp/likelihood.jsonl
oviq eval-likelihood --model dummy --backend dummy --dataset /tmp/likelihood.jsonl --out /tmp/likelihood.json
```

Compare a report against gates and render Markdown:

```bash
oviq compare --baseline /tmp/likelihood.json --current /tmp/likelihood.json \
  --gates configs/gates/default_gates.yaml --out /tmp/comparison.json
oviq report build --report /tmp/likelihood.json --out /tmp/likelihood-report --format all
oviq report render --bundle /tmp/likelihood-report --format markdown --out /tmp/likelihood.md
```

## OpenVINO export examples

Evaluation model for full-sequence logits:

```bash
optimum-cli export openvino \
  --model <HF_MODEL_ID_OR_PATH> \
  --task text-generation \
  --weight-format fp16 \
  ./models/<name>-ov-eval-fp16
```

Production model with KV cache:

```bash
optimum-cli export openvino \
  --model <HF_MODEL_ID_OR_PATH> \
  --task text-generation-with-past \
  --weight-format int8 \
  ./models/<name>-ov-prod-int8
```

OpenVINO GenAI generation is useful for output-level and smoke tests, but full
distribution metrics require OpenVINO Runtime, Optimum Intel, or an HF reference runner
that exposes logits.

## CLI commands

- `oviq eval-likelihood`: NLL/PPL over JSONL samples.
- `oviq eval-drift`: logits distribution drift between reference and current backends.
- `oviq eval-long-context`: context gain, saturation, position buckets and robustness checks.
- `oviq eval-serving`: batch invariance, generation prefix divergence and KV-cache interface.
- `oviq eval-rag`: rule-based retrieval, citation and faithfulness metrics with optional scorer hooks.
- `oviq eval-agent`: tool-use, grounding, state drift, completion and recovery metrics.
- `oviq run-suite`: suite config entry point.
- `oviq run-gpu-suite`: GPU metric verification scorecard for exported OpenVINO models.
- `oviq compare`: threshold gates over report metrics.
- `oviq report build/analyze/render/metrics-table/validate`: production report bundle,
  normalized metric observations, findings, Markdown, HTML and CSV artifacts.
- `oviq report reference-comparison`: compare reference-backed metrics across reports.
- `oviq metric-long-context`: utility for position-bucketed long-context metrics.
- `oviq list-genai-models`: inspect the recommended GenAI model matrix.
- `oviq list-metric-references`: inspect metric reference/oracle metadata used by reports and gates.
- `oviq genai-export-plan`: print OpenVINO export commands for a selected model.

## Metric families

- Inference equivalence: KL, JS, entropy drift, logit cosine, top-k overlap.
- Likelihood quality: NLL, PPL, sliding-window and bucketed PPL.
- Long context: context gain, saturation, lost-in-the-middle, distractor sensitivity.
- RAG and agent traces: component metrics and explicit placeholders for judge-based scores.
- Serving: batch invariance and KV-cache drift interfaces.

Every scalar metric intended for reporting or gating should have a registered
reference/oracle in `oviqs.domain.references`. Reports include a `metric_references` manifest,
and gates treat unreferenced metrics as `unknown`.

```bash
.venv/bin/oviq list-metric-references --family rag --json
```

## GenAI model matrix

Recommended GenAI models for metric testing are tracked in
`configs/examples/genai_metric_models.yaml` and documented in `docs/genai_model_matrix.md`.

```bash
.venv/bin/oviq list-genai-models --tier smoke --metric likelihood
.venv/bin/oviq genai-export-plan --model Qwen/Qwen2.5-0.5B-Instruct
```

## Documentation map

- [Security policy](SECURITY.md)
- [Architecture](docs/architecture.md)
- [Usage](docs/usage.md)
- [Data formats](docs/data_formats.md)
- [Metrics](docs/metrics.md)
- [Metric playbook](docs/metric_playbook.md)
- [OpenVINO runners](docs/openvino_runners.md)
- [GenAI model matrix](docs/genai_model_matrix.md)
- [Reports and gates](docs/reports_and_gates.md)
- [Long context](docs/long_context.md)
- [RAG and agent diagnostics](docs/rag_agent.md)
- [Integrations](docs/integrations.md)
- [CI/CD](docs/ci_cd.md)
- [Repo-local skills](docs/skills.md)
- [Remote GPU from scratch](docs/remote_gpu_from_scratch.md)
- [GPU requirements](docs/gpu_requirements.md)
