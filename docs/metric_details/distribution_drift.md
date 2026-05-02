# Distribution Drift Metrics

Distribution drift metrics compare the full next-token distribution from a reference run
and a current run at the same token positions. They are more sensitive than text-only
checks because the generated output can stay identical while probability mass moves among
secondary candidates.

Use them to answer:

- Did export, quantization, precision, device or runtime change the logits?
- Are differences global, tail-only or concentrated in a few positions?
- Can deterministic generation, RAG citation choice or agent tool choice change even if
  perplexity is still acceptable?

## Scientific Basis

OVIQS converts logits to probability distributions with `softmax`. For reference
distribution `P` and current distribution `Q`, Kullback-Leibler divergence is:

```text
KL(P || Q) = sum_i P_i * log(P_i / Q_i)
```

Jensen-Shannon divergence is a symmetric smoothed variant:

```text
M = 0.5 * (P + Q)
JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
```

Entropy drift measures uncertainty change:

```text
H(P) = -sum_i P_i * log(P_i)
entropy_drift = H(Q) - H(P)
```

Logit cosine compares raw logit vectors:

```text
cosine(z_ref, z_cur) = dot(z_ref, z_cur) / (||z_ref|| * ||z_cur||)
```

KL and JS operate in probability space; cosine and top-k overlap operate in ranking/logit
space. Use both when diagnosing regressions.

## OVIQS Implementation

Source: `src/oviqs/domain/metrics/distribution_drift.py`.

Required inputs:

- reference logits and current logits with identical shape ending in vocabulary dimension;
- identical tokenizer, prompt text, token IDs, padding, attention masks and position
  alignment;
- stable backend settings for deterministic comparison.

OVIQS converts both logit tensors to `float32`, applies `log_softmax`, computes per-position
KL, JS, reference entropy, current entropy, entropy drift and logit cosine, then aggregates
mean, percentile and max summaries.

## Metrics

### mean_kl

Formula:

```text
mean_kl = mean_t KL(P_ref_t || P_cur_t)
```

Direction: lower is better.

Interpretation:

- Near zero means the current distribution is close to reference across positions.
- A broad increase usually points to quantization, precision, model conversion or device
  plugin differences.
- KL is asymmetric; OVIQS intentionally asks how surprising the reference distribution is
  under the current distribution.

Action:

- Check the same samples with likelihood NLL.
- Compare FP32/FP16/INT8 separately to isolate precision effects.
- Inspect tokens with highest `kl_per_pos`.

### p95_kl

Formula:

```text
p95_kl = percentile_95({KL(P_ref_t || P_cur_t)})
```

Direction: lower is better.

Interpretation:

- Captures localized instability hidden by `mean_kl`.
- Useful for long prompts where a few positions can drive generation changes.
- If p95 regresses while mean is stable, inspect special tokens, sequence boundaries,
  newline-heavy prompts, code tokens and numeric tokens.

### max_kl

Formula:

```text
max_kl = max_t KL(P_ref_t || P_cur_t)
```

Direction: lower is better, but use mainly for debugging.

Interpretation:

- One pathological position can dominate the max.
- Do not use as a release gate unless the dataset is deterministic and stable.
- Use it to find exact sample and position for reproduction.

### mean_js

Formula:

```text
mean_js = mean_t JS(P_ref_t, P_cur_t)
```

Direction: lower is better.

Interpretation:

- JS is symmetric and bounded for a fixed log base, which makes it convenient for comparing
  several model variants against a reference.
- A model can have modest KL and JS but still fail top-1 checks if probability mass crosses
  the argmax boundary.

### p95_js

Formula:

```text
p95_js = percentile_95({JS(P_ref_t, P_cur_t)})
```

Direction: lower is better.

Interpretation:

- Tail-risk companion to `mean_js`.
- If p95 JS is high, inspect whether the same positions also have top-k overlap loss.

### mean_entropy_drift

Formula:

```text
mean_entropy_drift = mean_t (H(P_cur_t) - H(P_ref_t))
```

Direction: close to zero is best.

Interpretation:

- Positive value: current model is more uncertain or flatter.
- Negative value: current model is sharper or more overconfident.
- Entropy drift alone does not say whether the right tokens improved; combine with KL,
  NLL and task metrics.

### mean_logit_cosine

Formula:

```text
mean_logit_cosine = mean_t cosine(z_ref_t, z_cur_t)
```

Direction: higher is better, normally close to `1.0`.

Interpretation:

- Sensitive to vector direction in raw logit space.
- Low cosine with small KL can indicate scale/shift or ranking differences that are partly
  softened by softmax.
- Useful when debugging runtime output tensors before probability metrics.

### top10_overlap

Formula:

```text
top10_overlap = mean_t |Top10(z_ref_t) intersect Top10(z_cur_t)| / 10
```

Direction: higher is better.

Interpretation:

- Measures candidate-set stability.
- Low overlap means reranking changed enough that sampling, beam search or constrained
  decoding can produce different text.
- Use different `k` values for tool names, short labels or large vocabularies.

### top1_changed_rate

Formula:

```text
top1_changed_rate = mean_t [argmax(z_ref_t) != argmax(z_cur_t)]
```

Direction: lower is better.

Interpretation:

- Direct warning for deterministic decoding.
- Any non-zero value in a deterministic equivalence test deserves sample-level review.
- If top-1 changed but KL is tiny, the top candidates were probably very close; inspect
  margins before deciding severity.

## Dataset Design

Minimal drift dataset:

```json
{"id":"d1","task_type":"likelihood","text":"The release date is April 16, 2026."}
```

Use prompts that contain:

- numbers, dates, units and punctuation;
- code snippets and JSON;
- rare named entities;
- short labels or tool names;
- long contexts with retrieval boundaries.

Dataset rules:

- Keep reference and current token IDs identical.
- Record sample ID, token position and decoded token for every high-drift position when
  debugging.
- Use deterministic fixtures with tiny vocabularies in unit tests to validate formulas.
- Do not mix unrelated prompt families in one gate unless you also review per-family
  buckets.

## Decision Guide

| Observation | Likely cause | Action |
|---|---|---|
| Mean KL high across most samples | global export/precision shift | Compare FP32 vs FP16 vs INT8 and inspect output tensor names. |
| p95 or max high only | local instability | Decode worst positions and check special tokens, boundaries and masks. |
| Entropy drift positive | distribution flattened | Check quantization calibration and logits precision. |
| Entropy drift negative | distribution sharpened | Check overconfident INT8 path or logit scaling. |
| Top-1 changed but KL small | near tie at argmax | Inspect top-token margins and downstream deterministic output. |
| Top-k overlap low | candidate ranking changed | Run generation/RAG/agent fixtures before release. |

## References

- S. Kullback and R. A. Leibler, [On Information and Sufficiency](https://doi.org/10.1214/aoms/1177729694), Annals of Mathematical Statistics, 1951.
- J. Lin, [Divergence Measures Based on the Shannon Entropy](https://pdodds.w3.uvm.edu/research/papers/others/1991/lin1991a.pdf), IEEE Transactions on Information Theory, 1991.
- Claude Shannon, [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf), Bell System Technical Journal, 1948.
- scikit-learn, [cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html), for the standard vector-similarity definition.
