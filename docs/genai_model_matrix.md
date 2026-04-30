# GenAI Model Matrix For Metric Testing

This matrix defines the recommended OpenVINO GenAI-compatible model set for OVIQS metric
development. It is intentionally tiered: small models keep CI and local debugging cheap,
while larger and long-context models are reserved for scheduled or explicit quality runs.

Sources checked on 2026-04-27:

- [OpenVINO GenAI supported models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/)
- [OpenVINO GenAI inference guide](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-genai.html)
- [OpenVINO GenAI Python text-generation samples](https://openvinotoolkit.github.io/openvino.genai/docs/samples/python/text_generation/)

## Rules

1. Use OpenVINO GenAI models for generation, structured output, RAG, agent and serving
   metrics.
2. Use the same base model exported through Optimum/OpenVINO Runtime for logits-level
   metrics: NLL, PPL, KL, JS, entropy drift and logit cosine.
3. For each selected LLM, keep two OpenVINO exports when feasible:
   `text-generation` for full-sequence logits and `text-generation-with-past` for GenAI
   generation/KV-cache tests.
4. Do not use the target Intel GPU device file for ordinary unit tests. Use it only for
   explicit metric verification runs.
5. For requested GPU metric runs, use the exact requested model families. Do not replace
   `Qwen3.5` with `Qwen3`.

## Requested Minimal Non-Distilled Models

These are the target models for real GPU metric verification. Use the smallest official
non-distilled checkpoint or official compressed weight format when one is available.

| Family | Model | Minimal weight format | Notes |
|---|---|---|---|
| gpt-oss | `openai/gpt-oss-20b` | MXFP4/native compressed weights, export attempt as INT4 if needed | Smallest official gpt-oss model; 21B parameters with 3.6B active parameters. |
| Qwen 3.5 | `Qwen/Qwen3.5-0.8B` | INT4 OpenVINO export | Smallest official Qwen3.5 checkpoint found for this run. |
| Mistral | `mistralai/Ministral-3-3B-Instruct-2512` | official FP8, INT4 OpenVINO export if conversion supports it | Smallest Ministral 3 family model; non-distilled post-trained checkpoint. |
| Gemma 3 | `google/gemma-3-1b-it` | INT4 OpenVINO export | Smallest Gemma 3 instruction checkpoint. |

## Current Validated GPU Model

The current target GPU machine exposes only Intel UHD Graphics 770. On that device, the
validated non-toy model for end-to-end OVIQS metric verification is:

| Model | Export | Result |
|---|---|---|
| `openai-community/gpt2` | `text-generation` FP16 for logits, `text-generation-with-past` FP16 for GenAI | Passes likelihood, drift, long-context mechanics, serving batch drift and generation metrics on GPU. |

Use this model when validating OVIQS behavior on the current GPU server. The larger target
families remain useful compatibility targets, but they currently fail before producing
numeric GPU quality metrics on this hardware/software stack.

The extended GPT-2 validation run also downloads WikiText-2 validation samples and writes
`reports/target-models/gpt2_extended_gpu_metrics.json`. On the current Intel UHD Graphics
770 GPU it passes all implemented sections:

- WikiText-2 likelihood: NLL `3.6784854078497498`, PPL `39.58639140657846`
- CPU/GPU drift: mean KL `0.003599317201102773`, p95 KL `0.008398092041412989`, top-10 overlap `0.9663901909348929`
- FP16/INT8 drift: mean KL `0.03543248772621155`, p95 KL `0.07577071090539296`, top-10 overlap `0.8869679051632695`
- long-context sliding-window PPL `32.502554571259346`, lost-in-middle score `-0.05585379687722558`
- serving batch drift mean KL `0.0`, top-1 changed rate `0.0`
- RAG rule-based evidence coverage `1.0`; agent tool-call validity `1.0`

KV-cache drift requires a stateful cached-decode IR and is measured when the model exposes
OpenVINO inference state. Recovery-after-tool-error requires failure-trace inputs; runs keep
it explicit instead of fabricating values.

## Tier 0: Always-On Smoke Models

These are the first models to wire into scripts because they are small enough for frequent
local checks.

| Model | Why | Metrics |
|---|---|---|
| `sshleifer/tiny-gpt2` | Minimal CausalLM for GPU compile and end-to-end metric smoke. Not a quality model. | PPL, drift, long-context mechanics, GenAI smoke |
| `Qwen/Qwen2.5-0.5B-Instruct` | Small Qwen2 topology; good tokenizer and chat-template coverage. | PPL, drift, generation determinism, JSON validity |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Listed by OpenVINO GenAI and useful as a lightweight chat baseline. | PPL, generation repetition, smoke serving |
| `microsoft/Phi-3-mini-4k-instruct` | Small Phi3 family model with 4k context. | PPL, short context gain, structured generation |

Default first pick: `Qwen/Qwen2.5-0.5B-Instruct`.

## Tier 1: Main Development Models

Use these when validating that metrics behave consistently across model families.

| Model | Why | Metrics |
|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | Still manageable, stronger than 0.5B, useful for RAG/agent prompts. | PPL, drift, RAG answer quality, tool-call validity |
| `Qwen/Qwen2.5-3B-Instruct` | Better generation quality without jumping to 7B. | generation quality, JSON/schema compliance, RAG |
| `microsoft/Phi-4-mini-instruct` | OpenVINO GenAI samples verify it for compound grammar/tool-call style tests. | structured output, tool calls, agent traces |
| `meta-llama/Llama-3.2-3B-Instruct` | OpenVINO GenAI samples verify it for structural tag tool-call generation. | tool-call generation, structured output, agent metrics |

Default main pick: `Qwen/Qwen2.5-1.5B-Instruct`.

## Tier 2: Long-Context And Regression Models

Use these for scheduled runs, not for fast development loops.

| Model | Why | Metrics |
|---|---|---|
| `microsoft/Phi-3-mini-128k-instruct` | Explicit long-context Phi3 variant listed by OpenVINO GenAI. | sliding-window PPL, position buckets, lost-in-the-middle |
| `microsoft/Phi-3-medium-128k-instruct` | Larger long-context Phi3 variant for regression confirmation. | long-context degradation, distractor sensitivity |
| `Qwen/Qwen2.5-7B-Instruct` | Stronger Qwen baseline for quality comparisons. | RAG, generation quality, drift after quantization |

Default long-context pick: `microsoft/Phi-3-mini-128k-instruct`.

## RAG Support Models

For retrieval and reranking tests, keep these separate from LLM generation models.

| Model | Role | Metrics |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | Embeddings smoke model. | retrieval recall, context recall |
| `sentence-transformers/all-MiniLM-L12-v2` | Lightweight embedding baseline. | retrieval regression, token waste checks |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | Reranker baseline. | context precision, rank quality |

## Recommended Export Variants

For each LLM selected above:

```bash
optimum-cli export openvino \
  --model <HF_MODEL_ID_OR_PATH> \
  --task text-generation \
  --weight-format fp16 \
  ./models/<name>-ov-eval-fp16

optimum-cli export openvino \
  --model <HF_MODEL_ID_OR_PATH> \
  --task text-generation-with-past \
  --weight-format int8 \
  ./models/<name>-ov-genai-int8
```

Optional quantization matrix:

- `ov-eval-fp16`: reference OpenVINO full-forward logits.
- `ov-eval-int8`: logits-level quantization drift.
- `ov-genai-fp16-with-past`: generation/KV-cache baseline.
- `ov-genai-int8-with-past`: production-style generation and serving.
- `ov-genai-int4-with-past`: aggressive compression regression.

## Initial Run Order

1. `Qwen/Qwen2.5-0.5B-Instruct`
2. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
3. `Qwen/Qwen2.5-1.5B-Instruct`
4. `microsoft/Phi-4-mini-instruct`
5. `microsoft/Phi-3-mini-128k-instruct`

This order gives fast signal first and postpones long-context cost until the core metric
path is stable.

## CLI

List models suitable for a metric:

```bash
.venv/bin/oviq list-genai-models \
  --config configs/examples/genai_metric_models.yaml \
  --tier smoke \
  --metric likelihood
```

Generate export commands:

```bash
.venv/bin/oviq genai-export-plan \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --config configs/examples/genai_metric_models.yaml
```
