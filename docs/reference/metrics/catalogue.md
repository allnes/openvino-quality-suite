# Metric catalogue

Metric families:

- inference equivalence: distribution alignment, logit similarity and top-k overlap
- likelihood: NLL, perplexity and token-level probability diagnostics
- long context: position buckets, context gain, saturation and distractor checks
- generation: output-level quality signals when full logits are unavailable
- RAG: retrieval, citation and faithfulness-oriented diagnostics
- agent traces: tool use, grounding, state drift and recovery checks
- serving: batch invariance, prefix divergence and KV-cache interface checks
- performance: latency, throughput and resource-oriented observations

Gateable scalar metrics should have reference or oracle metadata. Reports expose
that metadata through `metric_references`, and missing unsupported metrics remain
`unknown` instead of being converted into pass/fail values.

## Family guide

| Family | Typical evidence | Use for | Missing-evidence status |
|---|---|---|---|
| Inference equivalence | Aligned logits from reference and current backend. | Export/runtime drift checks. | `unknown` when positions or tokenizer alignment differ. |
| Likelihood | Token log probabilities or equivalent likelihood data. | Regression checks over fixed prompts. | `unknown` when the backend cannot expose aligned log probabilities. |
| Long context | Position-aware likelihood or retrieval/generation evidence. | Context-window behavior. | `unknown` for lengths or positions that were not evaluated. |
| Generation | Produced text and structured output checks. | Output quality smoke tests. | `unknown` when judge labels or deterministic expectations are absent. |
| RAG | Questions, retrieved context, answers and citations. | Retrieval and grounding diagnostics. | `unknown` when retrieved context, answer labels or judge inputs are missing. |
| Agent traces | Tool calls, observations, final answer and expected state. | Agent workflow correctness. | `unknown` when the trace lacks expected tool/state evidence. |
| Serving | Single/batch paths, prefix behavior and cache behavior. | Production endpoint checks. | `unknown` when endpoint metadata or paired requests are unavailable. |
| Performance | Timing, throughput and resource observations. | Capacity and regression tracking. | `unknown` when timing/resource collection was not enabled. |

## Public metric paths

These paths are the current documentation contract for scalar rows and gates.
Use exact path names in `metrics.csv`, gate YAML and comparison output.

| Path | Family | Required evidence | Gate discipline |
|---|---|---|---|
| `likelihood.nll` | Likelihood | Token-aligned log probabilities and score mask. | Lower is better; compare against baseline or fixed maximum. |
| `likelihood.perplexity` | Likelihood | Token-aligned log probabilities over the same dataset slice. | Lower is better; gate only with a documented dataset and tokenizer. |
| `likelihood.num_tokens` | Likelihood | Count of scored tokens. | Use as context for other gates, not as a quality gate alone. |
| `distribution_drift.mean_kl` | Inference equivalence | Same-position reference/current probability distributions. | Lower is better; unknown when logits are not aligned. |
| `distribution_drift.mean_js` | Inference equivalence | Same-position reference/current probability distributions. | Lower is better; use with KL or top-k overlap. |
| `distribution_drift.mean_logit_cosine` | Inference equivalence | Same-position reference/current logits. | Higher is better; unknown for generation-only backends. |
| `distribution_drift.top1_changed_rate` | Inference equivalence | Top token per aligned position. | Lower is better; inspect sensitive tokens before blocking. |
| `distribution_drift.top5_overlap` | Inference equivalence | Top-k token sets per aligned position. | Higher is better; gate with tokenizer compatibility recorded. |
| `long_context.context_gain` | Long context | Matched short/long context runs. | Higher is better; compare only equivalent prompt families. |
| `long_context.lost_in_middle_score` | Long context | Position-bucketed quality or likelihood values. | Lower is better when score represents middle degradation. |
| `long_context.degradation_slope` | Long context | Quality values over ordered context lengths. | Lower degradation is better; gate on fixed length set. |
| `long_context.distractor_sensitivity` | Long context | Clean and distracted variants of the same samples. | Lower is better; unknown if paired samples are absent. |
| `rag.context_precision` | RAG | Retrieved contexts, question and relevant labels or judge. | Higher is better; unknown without retrieval evidence. |
| `rag.context_recall` | RAG | Retrieved contexts and expected supporting context. | Higher is better; unknown without ground truth. |
| `rag.faithfulness` | RAG | Answer, retrieved context and judge or deterministic checker. | Higher is better; record judge configuration. |
| `agent.tool_correctness` | Agent traces | Expected and actual tool calls. | Higher is better; unknown without expected trace labels. |
| `agent.argument_correctness` | Agent traces | Expected and actual tool arguments. | Higher is better; unknown when arguments are not structured. |
| `serving.batch_mean_kl` | Serving | Single and batched logits for identical prompts. | Lower is better; unknown when paired execution is absent. |
| `serving.prefix_divergence` | Serving | Prefix-equivalent requests and aligned outputs/logits. | Lower is better; record serving endpoint and cache settings. |
| `performance.latency_ms` | Performance | Timed request execution. | Lower is better; gate only within comparable hardware profiles. |
| `performance.tokens_per_second` | Performance | Token count and elapsed generation time. | Higher is better; gate only within comparable hardware profiles. |

## Backend availability

| Backend surface | Strong metrics | Common unknown metrics |
|---|---|---|
| OpenVINO Runtime logits path | likelihood, distribution drift, serving equivalence | generation-only judge scores without outputs |
| OpenVINO GenAI generation path | generation smoke, selected serving checks | aligned logits, KL/JS, token-level NLL |
| OVMS or remote serving endpoint | serving, latency, throughput, endpoint regressions | full-vocabulary drift unless logits are exposed |
| External RAG/agent scorers | RAG and agent trace metrics | deterministic likelihood and distribution metrics |
| Dummy backend | documentation fixtures and contract tests | production-quality model claims |

## Status semantics

Use these statuses consistently:

| Status | Meaning |
|---|---|
| `pass` | Evidence exists and the metric satisfies its configured oracle or gate. |
| `warning` | Evidence exists and the metric is degraded but not publication-blocking. |
| `fail` | Evidence exists and the metric violates a blocking gate. |
| `unknown` | Evidence, reference metadata or comparable baseline is missing. |

`unknown` is not a softer `pass`. It means the suite did not prove the check.

## Adding catalogue entries

When a new metric becomes public, update this page with:

- metric family;
- normalized path;
- required backend evidence;
- reference or oracle metadata when gateable;
- expected missing-evidence behavior.
