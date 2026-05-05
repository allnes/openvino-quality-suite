# OVIQS

OpenVINO Inference Quality Suite for LLM diagnostics.

OVIQS helps you evaluate logits-level quality, generation behavior, serving
stability, RAG outputs and agent traces, then publish versioned report bundles
for humans and automation.

Project status: pre-release. Report schemas are versioned contracts; Python
module paths may still change before the first stable release.

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
oviq report analyze --report /tmp/likelihood.json \
  --gates configs/gates/default_gates.yaml --out /tmp/likelihood-analysis.json
oviq report build --report /tmp/likelihood.json --out /tmp/likelihood-report --format all
oviq report render --bundle /tmp/likelihood-report --format markdown --out /tmp/likelihood.md
oviq report validate --report /tmp/likelihood-report/report.json
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
- `oviq compare`: compact baseline/current comparison JSON.
- `oviq report build`: build a report bundle with JSON, Markdown, HTML and CSV outputs.
- `oviq report analyze`: analyze a report against optional baseline and gates.
- `oviq report render`: render an existing report bundle.
- `oviq report metrics-table`: flatten scalar metrics to CSV.
- `oviq report validate`: validate an `EvaluationReport` JSON file.
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
`configs/examples/genai_metric_models.yaml` and exposed through the generated CLI
reference.

```bash
.venv/bin/oviq list-genai-models --tier smoke --metric likelihood
.venv/bin/oviq genai-export-plan --model Qwen/Qwen2.5-0.5B-Instruct
```

## Documentation map

- [Documentation portal](docs/index.md)
- [Security policy](SECURITY.md)
- [Quickstart](docs/start/quickstart.md)
- [Architecture](docs/explanation/architecture.md)
- [CLI reference](docs/reference/cli/index.md)
- [Reporting spec](docs/reference/reporting/reporting-spec.md)
- [JSON schemas](docs/reference/schemas/evaluation-report.md)
- [Metric playbook](docs/reference/metrics/playbook.md)
- [Metric catalogue](docs/reference/metrics/catalogue.md)
- [Developer guides](docs/developers/index.md)
- [Project governance](docs/project/contributing.md)
