# Architecture

OVIQS separates metric math from delivery mechanisms. The public package is organized as a
layered architecture:

1. Domain and metric layer: backend-independent sample, trace, report and metric contracts.
2. Application layer: evaluation use cases that turn requests into normalized reports.
3. Port layer: protocols for runners, datasets, reporting, storage, plugins and
   observability.
4. Adapter layer: concrete implementations for JSONL, local files, report rendering,
   entry-point plugins and model runners.
5. Interface layer: CLI, HTTP and gRPC boundaries that call application services.
6. Platform layer: composition, configuration profiles and path policy.

The diagnostic model still has two evaluation surfaces:

1. Logits-level inference diagnostics: HF, Optimum OpenVINO and OpenVINO Runtime runners expose
   logits for NLL, PPL and distribution drift.
2. Generation and application layer: OpenVINO GenAI, OVMS, RAG and agent traces are used
   for output-level diagnostics.

Metrics stay backend-agnostic once logits, generations or traces are available.

## Package layout

- `oviqs.domain`: report, sample, trace, metric and reference exports used by the
  application layer.
- `oviqs.application`: request DTOs, orchestration helpers and evaluation services.
- `oviqs.ports`: structural protocols for runners, datasets, reports, storage, plugins,
  tokenizers and observability.
- `oviqs.adapters`: concrete implementations for runner factories, JSONL datasets, report
  writers/renderers, local storage, plugin registries and observability sinks.
- `oviqs.interfaces`: CLI, HTTP and gRPC entry points.
- `oviqs.platform`: bootstrap container, profile configuration and path safety utilities.
- `oviqs.adapters.integrations`: optional external evaluation framework adapters.
- `oviqs.cli`: console entry module that exports the Typer app from `oviqs.interfaces.cli`.
- `oviqs.contracts`: JSON Schema and protobuf contracts for external report and worker
  integration.

This project is pre-stable. Legacy compatibility packages such as `oviqs.core`,
`oviqs.metrics`, `oviqs.runners`, `oviqs.datasets`, `oviqs.aggregation`,
`oviqs.reporting`, `oviqs.references` and `oviqs.models` are not public contracts and
must not be preserved as import shims. Code should move directly into the target layer.

## Data flow

1. An interface receives CLI, HTTP or gRPC input and builds an application request DTO.
2. The platform bootstrap container supplies ports for runners, datasets and reports.
3. Adapters load JSONL samples, rows or suite YAML and build the selected backend runner.
4. Application services produce logits, generations, traces or precomputed metric rows.
5. Domain-level metric functions compute backend-independent metrics.
6. Report writers emit a normalized `EvaluationReport` with `schema_version`.
7. Optional comparison and rendering services evaluate gates and write human-readable output.

The report schema is intentionally sectioned by diagnostic surface:
`inference_equivalence`, `likelihood`, `long_context`, `generation`, `rag`,
`agent`, `serving`, `performance` and `reproducibility`.

## Extension points

Package metadata declares entry-point groups for external plugins:

- `oviqs.runners`: plugin group for runner implementations that satisfy `LogitsRunnerPort` or
  `GenerationRunnerPort`.
- `oviqs.datasets`: plugin group for dataset adapters that satisfy `DatasetReaderPort`.
- `oviqs.reporters`: report IO or rendering adapters.

Built-in entry points currently expose the dummy runner, JSONL dataset adapter, JSON report
adapter and Markdown renderer. Keep entry-point implementations optional-dependency safe:
loading the package must not import OpenVINO, PyTorch, evaluator frameworks, service
clients or observability packages unless the selected adapter is instantiated.

## Configuration profiles

Profile YAML files under `configs/` provide public-safe defaults for common environments:

- `base.yaml`: shared suite defaults and strict reference/oracle gate policy.
- `local.yaml`: local CPU/dummy defaults.
- `ci.yaml`: CI defaults that can fail on unknown required metrics.
- `gpu.yaml`: OpenVINO Runtime GPU defaults.
- `prod.yaml`: production report defaults and strict gate policy.

Generated reports, model exports, local datasets, service credentials and operator notes
must stay outside these profiles. Environment-specific secrets belong in the runner
environment, not in repository configuration.

## Public contracts

`src/oviqs/contracts/jsonschema/evaluation_report.schema.json` defines the external JSON
report envelope. `src/oviqs/contracts/proto/evaluation.proto` defines the gRPC worker
request/response shape. These contracts are intentionally broad at the metric-section
level so new scalar diagnostics can be added without breaking existing consumers.

## Boundary between layers

Use logits-capable runners for teacher-forcing diagnostics such as NLL, PPL, KL,
JS, entropy drift and logit cosine. Use generation or serving adapters for output
quality, RAG, agent traces, smoke tests and production endpoint checks.

Do not infer full-distribution metrics from text-only generation outputs. If a
backend cannot expose aligned logits for the same token positions, record that
metric as missing or `unknown` instead of fabricating a comparable value.

Application services may import domain and port modules plus adapter factories supplied by
the bootstrap container. Interface and application code must not import legacy runner,
dataset, reporting or metric implementation modules. Tests enforce this boundary for
`oviqs.cli`, `oviqs.domain`, `oviqs.interfaces`, `oviqs.application` and `oviqs.ports`.

## GPU Metric Verification

GPU verification uses the same two-layer boundary:

- OpenVINO Runtime on `GPU` owns logits-level metrics: NLL, PPL, KL, JS, entropy drift,
  logit cosine, long-context bucketed PPL and batch invariance drift.
- OpenVINO GenAI on `GPU` owns generation-layer checks when a with-past GenAI export is
  available.
- RAG and agent checks are application-layer diagnostics. Rule-based checks can run
  locally from traces/fixtures; judge-based metrics must remain `unknown` unless an
  explicit scorer is configured.
