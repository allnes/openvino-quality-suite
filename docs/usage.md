# Usage

Install the project in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Use optional extras only for the surfaces you need:

```bash
python -m pip install -e ".[openvino,genai,eval,rag,agent,observability,plots]"
```

On a target GPU host, prefer the pinned requirements files before installing editable
extras:

```bash
python -m pip install -r requirements/gpu.txt
python -m pip install -r requirements/integrations.txt
python -m pip install -e ".[rag,agent,observability]" --no-deps
```

Install LightEval in a separate evaluator venv if Phoenix/Opik observability packages
are present:

```bash
python -m venv .venv-lighteval
.venv-lighteval/bin/python -m pip install -r requirements/lighteval.txt
```

## Likelihood

Create a JSONL dataset:

```bash
printf '{"id":"s1","task_type":"likelihood","text":"this is a test"}\n' > /tmp/likelihood.jsonl
```

Run with the deterministic dummy backend:

```bash
oviq eval-likelihood \
  --model dummy \
  --backend dummy \
  --dataset /tmp/likelihood.jsonl \
  --out /tmp/likelihood.json
```

For real model diagnostics, use `hf`, `optimum-openvino` or `openvino-runtime` and pass
the model path or ID through `--model`.

## Drift

Compare aligned reference and current logits:

```bash
oviq eval-drift \
  --reference ref-model \
  --current ov-model \
  --reference-backend hf \
  --current-backend optimum-openvino \
  --dataset /tmp/drift.jsonl \
  --out /tmp/drift.json
```

The command fails on logits shape mismatch because drift metrics require aligned token
positions and vocabulary dimensions.

## Long-context, serving, RAG and agent checks

Compute long-context metrics from JSONL samples:

```bash
oviq eval-long-context \
  --dataset /tmp/long_context.jsonl \
  --model dummy \
  --backend dummy \
  --lengths 4096,8192,16384 \
  --out /tmp/long_context.json
```

Run serving diagnostics with either default dummy samples or a serving JSONL dataset:

```bash
oviq eval-serving --model dummy --backend dummy --out /tmp/serving.json
```

Run rule-based RAG metrics with optional answers JSONL:

```bash
oviq eval-rag \
  --dataset /tmp/rag.jsonl \
  --answers /tmp/rag_answers.jsonl \
  --out /tmp/rag_report.json
```

Run agent trace metrics with optional expected-state/tool-schema rows:

```bash
oviq eval-agent \
  --traces /tmp/agent_traces.jsonl \
  --expected /tmp/agent_expected.jsonl \
  --out /tmp/agent_report.json
```

Judge-backed RAG and agent scores remain missing or `unknown` unless an external scorer
is explicitly configured and run.

## GPU metric suite

Run a multi-section GPU scorecard for an exported OpenVINO logits model:

```bash
oviq run-gpu-suite \
  --model models/sshleifer--tiny-gpt2-eval_logits \
  --backend openvino-runtime \
  --dataset /tmp/likelihood.jsonl \
  --device GPU \
  --out /tmp/gpu_suite.json
```

Add `--genai-model` when a `text-generation-with-past` OpenVINO GenAI export is available:

```bash
oviq run-gpu-suite \
  --model models/qwen--qwen2-5-0-5b-instruct-eval_logits \
  --genai-model models/qwen--qwen2-5-0-5b-instruct-genai-int4 \
  --dataset /tmp/likelihood.jsonl \
  --device GPU \
  --out /tmp/gpu_suite_with_genai.json
```

The suite records section-level failures in the report instead of aborting the full run
when a GPU compile or optional GenAI check fails.

## GenAI model matrix

List recommended models:

```bash
oviq list-genai-models --tier smoke --metric likelihood
```

Generate OpenVINO export commands:

```bash
oviq genai-export-plan \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --variant eval_logits \
  --variant genai_generation_int4
```

## Suite and gates

Run a suite scaffold from YAML:

```bash
oviq run-suite --config configs/default.yaml --out /tmp/suite.json
```

Evaluate gates:

```bash
oviq compare \
  --baseline /tmp/suite.json \
  --current /tmp/suite.json \
  --gates configs/gates/default_gates.yaml \
  --out /tmp/comparison.json
```

Inspect metric references before adding or tightening gates:

```bash
oviq list-metric-references
oviq list-metric-references --family serving --json
```

Reports include `metric_references` automatically. `compare` marks a gate check as
`unknown` when the metric has no registered reference/oracle, even if the numeric value
would otherwise pass the threshold.

Render a Markdown report:

```bash
oviq render-report --report /tmp/suite.json --out /tmp/suite.md
```

Build a reference-backed comparison across one or more reports:

```bash
oviq reference-comparison \
  --report baseline=/tmp/baseline.json \
  --report current=/tmp/current.json \
  --format markdown-transposed \
  --out /tmp/reference_comparison.md
```

Use `--all-metrics` to include every metric listed in report coverage, or
`--format html-dashboard` for a browsable dashboard artifact.

## Output locations

Use `/tmp`, CI artifact directories or the ignored local `reports/` path for generated
outputs. Do not commit generated model artifacts, caches or reports.
