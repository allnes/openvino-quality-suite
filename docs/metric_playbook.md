# Metric Playbook

This file is the table of contents and decision map for OVIQS metrics. Detailed metric
descriptions live in separate files under `docs/metric_details/`. Each detail file now
contains the metric purpose, mathematical definition, required inputs, example rows,
dataset guidance, interpretation rules, regression actions and scientific references.

## Scientific Reading Order

Use the metric families from lowest-level signal to highest-level application behavior:

| Layer | Family | What it isolates | Primary references |
|---|---|---|---|
| Token probability | [Likelihood](metric_details/likelihood.md) | Causal LM loss and perplexity before decoding. | [Shannon 1948](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf), [Transformers PPL guide](https://huggingface.co/docs/transformers/v4.32.0/perplexity) |
| Distribution equivalence | [Distribution Drift](metric_details/distribution_drift.md) | Full next-token distribution changes. | [Kullback-Leibler 1951](https://doi.org/10.1214/aoms/1177729694), [Lin 1991 JS divergence](https://pdodds.w3.uvm.edu/research/papers/others/1991/lin1991a.pdf) |
| Context scaling | [Long Context](metric_details/long_context.md) | Length, position and interference effects. | [Lost in the Middle](https://arxiv.org/abs/2307.03172), [RULER](https://arxiv.org/abs/2404.06654), [HELMET](https://arxiv.org/abs/2410.02694) |
| Decoded output | [Generation](metric_details/generation.md) | Text repetition, JSON validity and output contracts. | [BLEU](https://aclanthology.org/P02-1040/), [ROUGE](https://aclanthology.org/W04-1013/), [BERTScore](https://arxiv.org/abs/1904.09675) |
| Runtime equivalence | [Serving](metric_details/serving.md) | Batch, device and KV-cache consistency. | [OpenVINO State API](https://docs.openvino.ai/2023.3/openvino_docs_OV_UG_stateful_models_intro.html), [MLPerf Inference](https://arxiv.org/abs/1911.02549) |
| Retrieval grounding | [RAG](metric_details/rag.md) | Evidence, context precision/recall, citations and faithfulness. | [RAG](https://arxiv.org/abs/2005.11401), [RAGAS](https://arxiv.org/abs/2309.15217), [nDCG](https://sklearn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html) |
| Interactive traces | [Agent](metric_details/agent.md) | Tool calls, loops, state drift, grounding and recovery. | [ReAct](https://arxiv.org/abs/2210.03629), [AgentBench](https://arxiv.org/abs/2308.03688) |
| Runtime cost | [Performance](metric_details/performance.md) | Latency, tail latency and throughput. | [OpenVINO PerfMetrics](https://docs.openvino.ai/2025/api/genai_api/_autosummary/openvino_genai.PerfMetrics.html), [MLPerf Inference](https://arxiv.org/abs/1911.02549) |

Scientific use rule: never interpret an application-level metric in isolation. If RAG,
agent or generation quality changes, first check whether likelihood, distribution drift,
serving or long-context metrics already explain the regression.

## Decision Workflow

Use this order when diagnosing a model, export, quantization, device or serving change:

1. Check report status and section errors first. A failed compile or shape mismatch means
   the numeric metric path is not valid yet.
2. Check `metric_references`. A metric without a registered reference/oracle is not ready
   for quality gates.
3. Start with likelihood and distribution drift. These isolate logits-level changes before
   generation, RAG or agent behavior adds noise.
4. Check long-context and serving metrics next. These catch context-window, batching,
   device and KV-cache regressions.
5. Check generation, RAG and agent metrics last. These are application-level signals and
   often need rule-based fixtures or judge-backed labels.
6. Use gates only after the metric has a stable dataset, reference/oracle and expected
   direction.

Minimal diagnosis chain:

```text
status/errors
  -> metric_references
  -> num_tokens/data alignment
  -> likelihood NLL/PPL
  -> distribution drift
  -> serving and long-context equivalence
  -> generation/RAG/agent task metrics
  -> gates
```

Recommended first action by status:

| Status | Meaning | What to do |
|---|---|---|
| `pass` | Metric has a value, a reference/oracle and satisfies its gate. | Keep the report as a baseline candidate. |
| `warning` | Metric exists but violates a gate, or a section has expected unknowns. | Inspect sample-level metrics and compare with prior baseline. |
| `fail` | Metric section could not run, or an explicit failure was recorded. | Fix runner, model export, input shape, dataset or environment before interpreting quality. |
| `unknown` | Missing metric, missing reference, unsupported backend or unavailable scorer. | Decide whether to add data, add a reference/oracle, or keep the metric out of gates. |

## Metric Contents

### Likelihood

- [mean_log_prob](metric_details/likelihood.md#mean_log_prob)
- [nll](metric_details/likelihood.md#nll)
- [perplexity](metric_details/likelihood.md#perplexity)
- [num_tokens](metric_details/likelihood.md#num_tokens)
- [sliding_window_ppl](metric_details/likelihood.md#sliding_window_ppl)
- [position_bucketed_ppl](metric_details/likelihood.md#position_bucketed_ppl)

### Distribution Drift

- [mean_kl](metric_details/distribution_drift.md#mean_kl)
- [p95_kl](metric_details/distribution_drift.md#p95_kl)
- [max_kl](metric_details/distribution_drift.md#max_kl)
- [mean_js](metric_details/distribution_drift.md#mean_js)
- [p95_js](metric_details/distribution_drift.md#p95_js)
- [mean_entropy_drift](metric_details/distribution_drift.md#mean_entropy_drift)
- [mean_logit_cosine](metric_details/distribution_drift.md#mean_logit_cosine)
- [top10_overlap](metric_details/distribution_drift.md#top10_overlap)
- [top1_changed_rate](metric_details/distribution_drift.md#top1_changed_rate)

### Long Context

- [context_gain](metric_details/long_context.md#context_gain)
- [context_saturation_curve](metric_details/long_context.md#context_saturation_curve)
- [degradation_slope](metric_details/long_context.md#degradation_slope)
- [lost_in_middle_score](metric_details/long_context.md#lost_in_middle_score)
- [distractor_sensitivity](metric_details/long_context.md#distractor_sensitivity)
- [conflict_sensitivity](metric_details/long_context.md#conflict_sensitivity)
- [authoritative_margin](metric_details/long_context.md#authoritative_margin)
- [conflict_entropy](metric_details/long_context.md#conflict_entropy)

### Generation

- [repetition_rate](metric_details/generation.md#repetition_rate)
- [unique_ngram_ratio](metric_details/generation.md#unique_ngram_ratio)
- [json_valid](metric_details/generation.md#json_valid)
- [schema_validity](metric_details/generation.md#schema_validity)
- [required_section_coverage](metric_details/generation.md#required_section_coverage)
- [forbidden_section_violation](metric_details/generation.md#forbidden_section_violation)

### Serving

- [batch_invariance.mean_kl](metric_details/serving.md#batch_invariance_mean_kl)
- [batch_invariance.top1_changed_rate](metric_details/serving.md#batch_invariance_top1_changed_rate)
- [generation_prefix_divergence.prefix_divergence_rate](metric_details/serving.md#generation_prefix_divergence_prefix_divergence_rate)
- [kv_cache_drift.mean_kl](metric_details/serving.md#kv_cache_drift_mean_kl)
- [kv_cache_drift.top1_changed_rate](metric_details/serving.md#kv_cache_drift_top1_changed_rate)

### RAG

- [evidence_coverage](metric_details/rag.md#evidence_coverage)
- [context_precision](metric_details/rag.md#context_precision)
- [context_recall](metric_details/rag.md#context_recall)
- [citation_precision](metric_details/rag.md#citation_precision)
- [citation_recall](metric_details/rag.md#citation_recall)
- [faithfulness](metric_details/rag.md#faithfulness)
- [distractor_ratio](metric_details/rag.md#distractor_ratio)
- [supported_claim_ratio](metric_details/rag.md#supported_claim_ratio)

### Agent

- [tool_call_validity](metric_details/agent.md#tool_call_validity)
- [redundant_tool_call_rate](metric_details/agent.md#redundant_tool_call_rate)
- [agent_state_drift / state_drift_score](metric_details/agent.md#agent_state_drift)
- [observation_grounding_score](metric_details/agent.md#observation_grounding_score)
- [task_completion](metric_details/agent.md#task_completion)
- [policy_violation_rate](metric_details/agent.md#policy_violation_rate)
- [recovery_after_tool_error](metric_details/agent.md#recovery_after_tool_error)

### Performance

- [forward_latency_ms_mean](metric_details/performance.md#forward_latency_ms_mean)
- [forward_latency_ms_p95](metric_details/performance.md#forward_latency_ms_p95)
- [tokens_per_second_forward](metric_details/performance.md#tokens_per_second_forward)
- [generation_latency_ms](metric_details/performance.md#generation_latency_ms)

## Common Inputs

Minimal likelihood JSONL:

```json
{"id":"s1","task_type":"likelihood","text":"OpenVINO runs language model inference on Intel GPU."}
```

Structured long-context sample:

```json
{
  "id": "lc1",
  "task_type": "long_context",
  "context": "Long controlled context with facts and distractors.",
  "target": "The target answer or continuation.",
  "metadata": {
    "clean_nll": 2.1,
    "distracted_nll": 2.7,
    "conflict_nll": 3.2,
    "candidate_logprobs": {"Paris": -0.4, "Berlin": -3.0},
    "authoritative_key": "Paris"
  }
}
```

RAG sample plus answer row:

```json
{"id":"r1","task_type":"rag","prompt":"When was the release?","retrieved_contexts":["Doc A says April 16.","Noise"],"expected_evidence":["April 16"],"references":["doc-a"],"metadata":{"relevant_context_indices":[0]}}
```

```json
{"id":"r1","answer":"April 16","claims":["April 16"],"citations":["doc-a"]}
```

Agent trace plus expected row:

```json
{"id":"a1","input":"find date","steps":[{"type":"tool_call","tool":"search","args":{"query":"date"}},{"type":"observation","result":"April 16"},{"type":"final","content":"April 16"}]}
```

```json
{"id":"a1","tool_schemas":{"search":{"required":["query"]}},"expected_state":{"done":true},"actual_state":{"done":true}}
```

## Gates and Actions

Gate names encode direction:

- `<metric>_max`: value must be less than or equal to threshold.
- `<metric>_min`: value must be greater than or equal to threshold.
- `abs_<metric>_max`: absolute value must be within threshold.

Before adding a gate:

1. Register or confirm the metric reference with `oviq list-metric-references`.
2. Run the metric on a deterministic fixture.
3. Run it on a representative dataset.
4. Compare at least one known-good and one known-bad run.
5. Add a threshold only after the metric direction and noise are understood.

Example:

```yaml
inference_equivalence:
  mean_kl_max: 0.01
  p95_kl_max: 0.05
  mean_logit_cosine_min: 0.995

rag:
  context_recall_min: 0.90
  distractor_ratio_max: 0.25
```

If a gate returns `unknown`, decide whether the metric is missing, the metric path is
wrong, or the metric has no registered reference/oracle. Do not treat `unknown` as pass.
