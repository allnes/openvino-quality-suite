# Architecture

OVIQS has two layers:

1. Logits-level inference core: HF, Optimum OpenVINO and OpenVINO Runtime runners expose
   logits for NLL, PPL and distribution drift.
2. Generation and application layer: OpenVINO GenAI, OVMS, RAG and agent traces are used
   for output-level diagnostics.

Metrics stay backend-agnostic once logits, generations or traces are available.

## Package layout

- `oviqs.core`: shared schemas for samples, traces, reports, runners and metrics.
- `oviqs.metrics`: backend-independent metric implementations.
- `oviqs.runners`: runner adapters that expose logits or generation behavior.
- `oviqs.datasets`: JSONL loaders and dataset-specific adapters.
- `oviqs.aggregation`: gates, bucketing, statistics, comparisons and scorecards.
- `oviqs.reporting`: JSON, Markdown, HTML and plot report renderers.
- `oviqs.integrations`: optional adapters for external evaluation frameworks.
- `oviqs.cli`: Typer command surface.

## Data flow

1. Load JSONL samples or suite YAML.
2. Build a runner for the selected backend.
3. Produce logits, generations, traces or precomputed metrics.
4. Compute backend-independent metrics.
5. Write a normalized `EvaluationReport`.
6. Optionally compare the report against gates and render a human-readable output.

The report schema is intentionally sectioned by diagnostic surface:
`inference_equivalence`, `likelihood`, `long_context`, `generation`, `rag`,
`agent`, `serving`, `performance` and `reproducibility`.

## Boundary between layers

Use logits-capable runners for teacher-forcing diagnostics such as NLL, PPL, KL,
JS, entropy drift and logit cosine. Use generation or serving adapters for output
quality, RAG, agent traces, smoke tests and production endpoint checks.

Do not infer full-distribution metrics from text-only generation outputs. If a
backend cannot expose aligned logits for the same token positions, record that
metric as missing or `unknown` instead of fabricating a comparable value.

## GPU Metric Verification

GPU verification uses the same two-layer boundary:

- OpenVINO Runtime on `GPU` owns logits-level metrics: NLL, PPL, KL, JS, entropy drift,
  logit cosine, long-context bucketed PPL and batch invariance drift.
- OpenVINO GenAI on `GPU` owns generation-layer checks when a with-past GenAI export is
  available.
- RAG and agent checks are application-layer diagnostics. Rule-based checks can run
  locally from traces/fixtures; judge-based metrics must remain `unknown` unless an
  explicit scorer is configured.
