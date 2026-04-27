# Integrations

Optional integrations are implemented as lazy Python adapters for LightEval,
lm-evaluation-harness, OpenCompass, Ragas, DeepEval, Phoenix, Opik, OpenEvals,
agentevals and promptfoo result import.

## Current adapter surfaces

- `base.py`: shared `IntegrationResult`, optional dependency handling and callable
  execution normalization.
- `lighteval_adapter.py`: native Python pipeline adapter using `pipeline.evaluate(...)`
  or `pipeline.run(...)`.
- `lm_eval_adapter.py`: native `lm_eval.simple_evaluate(...)` wrapper.
- `opencompass_adapter.py`: native callable runner boundary and JSON report importer.
- `ragas_adapter.py`: Ragas dataset row builder and `ragas.evaluate(...)` wrapper.
- `deepeval_adapter.py`: `LLMTestCase` builder and `deepeval.evaluate(...)` wrapper.
- `phoenix_adapter.py` and `opik_adapter.py`: agent trace to span conversion plus
  Python client export adapters.
- `openevals_adapter.py`: OpenEvals/agentevals callable evaluator adapters.
- `promptfoo_adapter.py`: promptfoo JSON result importer and explicit Python callable
  adapter. OVIQS does not silently shell out to the promptfoo CLI.

Benchmark dataset adapters:

- `datasets/longbench.py`: LongBench row to `EvalSample` conversion.
- `datasets/helmet.py`: HELMET row to `EvalSample` conversion.
- `datasets/ruler.py`: RULER row to `EvalSample` conversion.

## Integration rules

Keep external frameworks optional. Core imports must not require heavyweight extras
unless the corresponding adapter is explicitly used.

Normalize outputs into OVIQS report sections instead of leaking framework-specific
schemas through the public report contract. If an external evaluator needs a judge
model or service credentials, make that dependency explicit and leave the metric
missing when it is unavailable.

Subprocess execution is not the default integration mode. If a framework exposes only a
CLI in a given installation, the adapter should return `unavailable` or import a result
file produced by that tool. A CLI backend can be added later as an explicit opt-in, but
adapters must not launch shell commands implicitly.

## Target GPU installation

For a target GPU machine, install the OpenVINO runtime stack first and then the
external evaluator stack:

```bash
python -m pip install -r requirements/gpu.txt
python -m pip install -r requirements/integrations.txt
python -m pip install -e ".[rag,agent,observability]" --no-deps
```

The integration requirements file pins the evaluator front doors and the transitive
resolver anchors needed by Phoenix and Opik. Proxy settings are environment-specific
and must be supplied by the runner environment, not committed to repo docs.

LightEval should be installed in a separate evaluator venv when Phoenix/Opik are also
installed on the same target. Current LightEval and observability stacks require
incompatible transitive ranges for protobuf/botocore:

```bash
python -m venv .venv-lighteval
.venv-lighteval/bin/python -m pip install -r requirements/lighteval.txt
```
