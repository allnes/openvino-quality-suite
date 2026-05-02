# Long-Context Metrics

Long-context metrics evaluate whether a model uses additional context reliably as sequence
length grows. They are essential for OpenVINO export and serving validation because long
contexts stress attention masks, position IDs, RoPE scaling, dynamic shapes, memory layout
and KV-cache behavior.

Use them to answer:

- Does more context improve target likelihood or task quality?
- At what length does quality saturate or degrade?
- Does the model ignore evidence in the middle of a long prompt?
- Are distractors or conflicting sources making the answer less likely?

## Scientific Basis

Long-context evaluation separates three effects:

- length effect: quality as a function of total context length;
- position effect: quality as a function of where relevant evidence appears;
- interference effect: quality under distractors or contradictory evidence.

OVIQS supports likelihood-style long-context metrics, where lower NLL/PPL indicates the
target is easier for the model, and quality-style metrics, where higher task score
indicates better answer accuracy. Direction matters for every metric.

For NLL-based comparisons, useful context should reduce loss:

```text
context_gain(length) = NLL(baseline) - NLL(length)
```

Positive gain means the extra context helped. Negative gain means it hurt.

## OVIQS Implementation

Source: `src/oviqs/domain/metrics/long_context.py`.

Core functions:

- `context_gain(nll_by_context, baseline_key="0k")`;
- `context_saturation_curve(nll_by_context, baseline_length=0)`;
- `lost_in_middle_score_from_ppl(position_bucketed_ppl)`;
- `lost_in_middle_score_from_quality(position_score)`;
- `degradation_slope(length_to_quality)`;
- `distractor_sensitivity(nll_clean, nll_distracted)`;
- `conflict_sensitivity(nll_clean, nll_conflicted)`;
- `authoritative_margin(candidate_logprobs, authoritative_key)`;
- `conflict_entropy(candidate_logprobs)`.

## Metrics

### context_gain

Formula:

```text
context_gain_k = NLL_baseline - NLL_k
```

Direction: higher is better.

Interpretation:

- Positive: context made the target more likely.
- Near zero: added context did not matter or the model could not use it.
- Negative: context hurt the target, often due to distractors, truncation, prompt packing
  or position handling.

Action:

- If gain is negative only after export, compare logits at each length.
- If gain is negative in both reference and current models, inspect retrieval/context
  construction rather than OpenVINO.

### context_saturation_curve

Definition:

```text
for each length:
  nll(length)
  ppl(length) = exp(nll(length))
  context_gain(length) = NLL_baseline - NLL(length)
```

Direction: NLL/PPL lower is better; gain higher is better.

Interpretation:

- Healthy curve: improves with more useful context, then saturates.
- Saturation is expected when additional tokens add little information.
- Degradation at longer lengths indicates packing, position, attention-mask, truncation or
  model-context mismatch.

Action:

- Plot by prompt family and length bucket.
- Compare `window_size` and `stride` for sliding-window PPL.
- Check exact max sequence length supported by model artifact and runtime.

### degradation_slope

Formula:

```text
quality = a * log2(length) + b
degradation_slope = a
```

Direction: higher is better when `quality` is an accuracy-style score.

Interpretation:

- Positive slope: quality improves as context grows.
- Near zero: quality is length-insensitive.
- Negative slope: quality degrades as context grows.

Important: do not use this formula with NLL unless you invert direction or document that
lower is better.

### lost_in_middle_score

For PPL buckets:

```text
edges = mean(PPL_0_10, PPL_90_100)
middle = mean(PPL_30_50, PPL_50_70)
lost_in_middle_score = middle / edges - 1
```

Direction: closer to `0` is better; positive means middle is worse.

For quality buckets:

```text
edges = mean(score_0_10, score_90_100)
middle = mean(score_30_50, score_50_70)
lost_in_middle_score = 1 - middle / edges
```

Interpretation:

- Positive score means evidence in the middle is harder to use than evidence near edges.
- This is a placement metric, not a pure model-quality metric; retrieval ordering can
  change it without changing model weights.

Action:

- Put high-value evidence earlier or later in the prompt.
- Reduce top-k or rerank to avoid burying relevant chunks.
- Test with controlled synthetic positions before changing production retrieval.

### distractor_sensitivity

Formula:

```text
distractor_sensitivity = NLL(distracted) - NLL(clean)
```

Direction: lower is better; `0` means no effect.

Interpretation:

- Positive: distractors made the target less likely.
- Negative: distracted prompt accidentally made the target easier.
- High sensitivity usually points to retrieval noise, prompt framing or long-context
  interference.

### conflict_sensitivity

Formula:

```text
conflict_sensitivity = NLL(conflicted) - NLL(clean)
```

Direction: lower is better only when the authoritative target remains unchanged.

Interpretation:

- Positive: conflicting evidence hurt the correct target.
- Negative: labels or prompt design may be ambiguous, or the conflict text made the target
  easier by repeating related words.
- Always inspect `authoritative_key` and source labels.

### authoritative_margin

Formula:

```text
authoritative_margin =
  logprob(authoritative_answer) - max(logprob(conflicting_answer_i))
```

Direction: higher is better.

Interpretation:

- Positive: authoritative answer is more likely than any conflicting candidate.
- Near zero: model is uncertain between sources.
- Negative: model prefers a conflicting answer.

Action:

- Improve source priority instructions.
- Add citation/source labels.
- Reduce conflicting context when product behavior requires one authoritative source.

### conflict_entropy

Formula over candidate answer probabilities:

```text
p_i = exp(logprob_i) / sum_j exp(logprob_j)
conflict_entropy = -sum_i p_i * log(p_i)
```

Direction: context-dependent. Lower means more decisive; too low with a wrong margin means
confidently wrong.

Interpretation:

- High entropy: model is uncertain among candidates.
- Low entropy with positive authoritative margin: decisive and likely correct.
- Low entropy with negative authoritative margin: decisive but following the wrong source.

## Dataset Design

Controlled long-context row:

```json
{
  "id": "lc1",
  "task_type": "long_context",
  "context": "Long context with relevant fact, distractors and optional conflict.",
  "target": "The authoritative answer is Paris.",
  "metadata": {
    "length_bucket": 8192,
    "evidence_position_bucket": "50_70",
    "clean_nll": 2.1,
    "distracted_nll": 2.7,
    "conflict_nll": 3.2,
    "candidate_logprobs": {"Paris": -0.4, "Berlin": -3.0},
    "authoritative_key": "Paris"
  }
}
```

Recommended datasets:

- RULER-style synthetic retrieval and variable-length tasks;
- LongBench-style real long-context QA/summarization/multi-document tasks;
- HELMET-style application-centric tasks up to long lengths;
- production RAG contexts with known relevant and distractor passages;
- controlled conflict fixtures with explicit source authority labels.

Dataset rules:

- Stratify by length, evidence position and task type.
- Keep answer labels independent from prompt wording when possible.
- Include clean, distracted and conflicted variants of the same base sample.
- Record tokenized length after the exact tokenizer used by the model.
- Do not compare long-context scores across models with different truncation behavior
  unless truncation is explicitly part of the evaluation.

## Decision Guide

| Observation | Likely cause | Action |
|---|---|---|
| Context gain positive then flat | useful context saturates | Keep shorter context if latency matters. |
| Context gain becomes negative at long lengths | interference or context handling bug | Check packing, masks, position IDs and retrieval noise. |
| Lost-in-middle high | evidence placement problem | Rerank or place high-value chunks at prompt edges. |
| Distractor sensitivity high | retriever returns noisy chunks | Tune top-k, chunking and reranking. |
| Authoritative margin negative | model follows wrong source | Strengthen source hierarchy and citation instructions. |
| Conflict entropy high | ambiguity among candidates | Add clearer labels or reduce conflicting context. |

## References

- Nelson F. Liu et al., [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172), TACL 2024.
- Yushi Bai et al., [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508), 2023.
- Cheng-Ping Hsieh et al., [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654), 2024.
- Jinhyuk Yen et al., [HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly](https://arxiv.org/abs/2410.02694), 2024.
