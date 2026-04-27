# Metrics

This page is the compact metric reference. For scientific definitions, formulas, examples,
datasets, interpretation rules, regression actions and source links, use
[Metric Playbook](metric_playbook.md) and the per-family detail pages:
[Likelihood](metric_details/likelihood.md),
[Distribution Drift](metric_details/distribution_drift.md),
[Long Context](metric_details/long_context.md),
[Generation](metric_details/generation.md),
[Serving](metric_details/serving.md),
[RAG](metric_details/rag.md),
[Agent](metric_details/agent.md) and
[Performance](metric_details/performance.md).

NLL is `-mean log p(x_t | x_<t)`. PPL is `exp(NLL)`. Bucketed PPL aggregates token NLL
first and exponentiates the mean.

Distribution drift computes `KL(P_ref || P_cur)`, JS divergence, entropy drift and logit
cosine on aligned logits.

Long-context diagnostics include context gain, saturation curves, lost-in-the-middle
score, distractor sensitivity, conflict sensitivity, conflict entropy and authoritative
answer margin.

## Likelihood

Likelihood metrics require target-token probabilities from aligned logits:

- `nll`: mean negative log-likelihood over evaluated tokens.
- `perplexity`: `exp(nll)`.
- `num_tokens`: number of evaluated target tokens.
- Sliding-window PPL: compute per-window token losses, aggregate token losses first,
  then exponentiate once.

## Distribution drift

Drift metrics compare reference and current logits at the same token positions:

- `kl`: `KL(P_ref || P_cur)`.
- `js`: Jensen-Shannon divergence.
- `entropy_ref`, `entropy_cur` and `entropy_drift`.
- `logit_cosine`: cosine similarity of raw logits.
- `top_k_overlap`: overlap in top candidates when requested by the metric function.

Shape mismatches must fail fast because they indicate unaligned tokenization or runner
behavior.

## Reference and Degradation Oracles

Every metric used in a report or gate must have a reference/oracle that can decide whether
the current result degraded. The runtime catalog lives in `oviqs.references` and is included
in JSON reports as `metric_references`.

Use the CLI to inspect the catalog:

```bash
oviq list-metric-references --json
oviq list-metric-references --family rag
```

Recommended primary references:

| Family | Reference/oracle |
|---|---|
| `likelihood` | Shannon information theory, LightEval corpus PPL, HF fixed-length PPL guidance and deterministic logits fixtures. |
| `distribution_drift` | Kullback-Leibler divergence, Jensen-Shannon divergence, entropy and deterministic full-vocabulary logits fixtures. |
| `long_context` | Lost-in-the-Middle, RULER, LongBench and HELMET-style controlled length, position, distractor and conflict datasets. |
| `generation` | BLEU/ROUGE/BERTScore context for text evaluation plus deterministic JSON/schema/entity fixtures. |
| `serving` | Same-prompt invariance oracle: alone vs batch, full-forward vs KV-cache, device/precision baseline vs current, aligned with OpenVINO stateful-model behavior. |
| `rag` | RAGAS, classical IR precision/recall/nDCG, SentenceTransformers retrieval evaluation and Phoenix-style retrieval/response separation. |
| `agent` | ReAct/AgentBench evaluation framing, DeepEval Tool Correctness, Task Completion and deterministic trace fixtures. |
| `performance` | MLPerf/OpenVINO latency-throughput methodology plus fixed-shape runner measurements. |

If a metric has no registered reference, treat it as not ready for quality gating.

## Gates

Gate names encode threshold direction:

- `<metric>_max`: pass when metric is less than or equal to the threshold.
- `<metric>_min`: pass when metric is greater than or equal to the threshold.
- `abs_<metric>_max`: pass when the absolute metric value is within threshold.

Missing metrics produce `unknown`; exceeded thresholds produce `warning`.
Metrics without a registered reference/oracle also produce `unknown` and must not be
treated as a pass.

## Long Context

Long-context metrics are exposed through `oviq eval-long-context` and `oviq run-gpu-suite`:

- `context_gain`: baseline NLL minus NLL at a longer context length.
- `degradation_slope`: linear slope of quality over log2 context length.
- `position_bucketed_ppl`: token-level NLL grouped by relative position buckets.
- `lost_in_middle_score`: middle bucket PPL relative to edge bucket PPL.
- `distractor_sensitivity`: `NLL(with_distractors) - NLL(clean)` when dataset metadata
  provides clean/distracted NLL values.
- `conflict_sensitivity`: `NLL(with_conflict) - NLL(clean)` when dataset metadata provides
  conflict NLL values.
- `authoritative_margin`: log-probability margin between authoritative and conflicting
  candidate answers.
- `conflict_entropy`: entropy across candidate answer probabilities.

## Serving

Serving diagnostics are exposed through `oviq eval-serving` and `oviq run-gpu-suite`:

- `batch_invariance`: KL, JS, entropy drift, logit cosine, top-k overlap and top-1 change
  rate between a prompt run alone and the same prompt inside a batch.
- `generation_prefix_divergence`: deterministic text prefix mismatch rate when a generation
  runner is available.
- `kv_cache_drift`: compares full-forward logits with one-token cached decode logits from a
  stateful OpenVINO IR. It is reported as `null` only when the supplied model has no exposed
  OpenVINO inference state.

## RAG

RAG metrics are exposed through `oviq eval-rag`:

- `evidence_coverage`: fraction of expected evidence strings found in retrieved contexts.
- `context_precision`: fraction of retrieved contexts containing expected evidence.
- `context_recall`: fraction of expected evidence covered by retrieved contexts.
- `citation_precision` and `citation_recall`: exact citation overlap metrics.
- `faithfulness`: conservative rule-based literal claim coverage.
- `distractor_ratio`: fraction of retrieved contexts not marked relevant.
- `supported_claim_ratio`: explicitly `unknown` unless an external judge/scorer is configured.

## Agent

Agent trace metrics are exposed through `oviq eval-agent`:

- `tool_call_validity`: schema-based validity of tool calls.
- `redundant_tool_call_rate`: repeated tool+args calls within the trace.
- `agent_state_drift`: mismatch rate between expected and actual state keys.
- `observation_grounding_score`: rule-based literal grounding of final/message text in
  observations.
- `task_completion`: whether the trace reaches a final answer without a later error.
- `policy_violation_rate`: fraction of steps marked with `metadata.policy_violation`.
- `recovery_after_tool_error`: fraction of tool errors followed by another tool call or final
  answer.
