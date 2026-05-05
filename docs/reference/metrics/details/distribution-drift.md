# Distribution drift metrics

Distribution drift metrics compare a reference token distribution with a current
token distribution at identical sample and token positions. They are runtime and
export diagnostics, not text-generation quality scores.

## Metrics

| Metric | Definition | Direction | Required evidence |
|---|---|---|---|
| `kl_per_pos` | KL divergence from the reference distribution to the current distribution at each aligned position. | Lower is better. | Full-vocabulary reference and current logits. |
| `mean_kl` | Mean of `kl_per_pos`. | Lower is better. | Same as `kl_per_pos`. |
| `p95_kl` | 95th percentile of `kl_per_pos`. | Lower is better. | Same as `kl_per_pos`. |
| `max_kl` | Maximum per-position KL. | Lower is better; inspect outliers. | Same as `kl_per_pos`. |
| `js_per_pos` | Jensen-Shannon divergence at each aligned position. | Lower is better. | Full-vocabulary distributions. |
| `mean_js` | Mean JS divergence. | Lower is better. | Same as `js_per_pos`. |
| `p95_js` | Tail JS divergence. | Lower is better. | Same as `js_per_pos`. |
| `ref_entropy_per_pos` | Shannon entropy of the reference distribution. | Context metric. | Reference distribution. |
| `cur_entropy_per_pos` | Shannon entropy of the current distribution. | Context metric. | Current distribution. |
| `entropy_drift_per_pos` | `cur_entropy - ref_entropy` per position. | Near zero is best. | Both distributions. |
| `mean_entropy_drift` | Mean entropy change. | Near zero is best. | Both distributions. |
| `logit_cosine_per_pos` | Cosine similarity of raw logit vectors per position. | Higher is better. | Same-shape logits. |
| `mean_logit_cosine` | Mean logit cosine. | Higher is better. | Same-shape logits. |
| `top1_changed_rate` | Fraction of positions where argmax token changed. | Lower is better. | Aligned logits. |
| `top5_overlap` | Mean overlap of top-5 token sets. | Higher is better. | Aligned logits. |
| `top10_overlap` | Mean overlap of top-10 token sets. | Higher is better. | Aligned logits. |
| `topk_overlap` | Mean overlap for configured `k`. | Higher is better. | Aligned logits and configured `k`. |
| `target_rank_delta` | Change in target-token rank. | Lower absolute drift is better. | Target token id and ranked distributions. |
| `sensitive_token_drift` | Drift on a configured sensitive token set. | Lower is better. | Sensitive token ids and aligned distributions. |

## Interpretation

KL is asymmetric and uses the reference distribution as the source of truth.
JS is symmetric and bounded, so it is useful when reviewers need a more stable
summary. Top-k overlap explains whether drift affects likely decoding choices.
Entropy drift explains confidence changes even when the top token is unchanged.

## Dataset examples

Use deterministic logits fixtures for unit tests and exported OpenVINO Runtime
models against an HF or Optimum reference for integration checks. Every row must
record tokenizer identity, model artifact identity, sample id and exact token
position alignment.

## Action policy

If shapes, tokenizer ids or positions do not align, do not compare the values.
Investigate alignment first. For real drift, inspect tail positions, sensitive
tokens and top-k changes before deciding whether a numerical drift is
release-blocking.
