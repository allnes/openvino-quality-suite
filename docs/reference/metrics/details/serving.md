# Serving metrics

Serving metrics compare execution modes for the same prompt and model artifact:
single versus batch, full-forward versus KV cache, or device and precision
variants. They detect production instability that aggregate quality metrics can
miss.

## Metrics

| Metric | Definition | Direction | Required evidence |
|---|---|---|---|
| `batch_invariance` | Summary object for single-vs-batch serving drift. | Review child fields. | Paired single and batch outputs/logits. |
| `batch_invariance_drift` | Distribution drift caused by batching. | Lower is better. | Paired logits. |
| `batch_invariance_mean_kl` | Mean KL for single-vs-batched logits. | Lower is better. | Paired logits. |
| `batch_mean_kl` | Mean KL for single-vs-batched inference. | Lower is better. | Paired logits. |
| `batch_p95_kl` | Tail KL for single-vs-batched inference. | Lower is better. | Paired logits. |
| `batch_js` | Mean JS for single-vs-batched logits. | Lower is better. | Paired logits. |
| `batch_entropy_drift` | Mean entropy shift introduced by batching. | Near zero is best. | Paired logits. |
| `batch_top1_changed_rate` | Top-1 instability introduced by batching. | Lower is better. | Paired logits. |
| `batch_generation_prefix_divergence` | Prefix divergence for single-vs-batch generation. | Lower is better. | Paired deterministic generations. |
| `generation_prefix_divergence` | Prefix divergence between deterministic generations. | Lower is better. | Two deterministic outputs. |
| `prefix_divergence_rate` | Share of generated prefix that diverged. | Lower is better. | Token-aligned prefixes. |
| `kv_cache_drift` | Summary object for full-forward vs cached decode drift. | Review child fields. | Paired full/cache logits. |
| `kv_cache_mean_kl` | Mean KL for full-forward vs cached decode. | Lower is better. | Paired logits. |
| `kv_cache_p95_kl` | P95 KL for full-forward vs cached decode. | Lower is better. | Paired logits. |
| `kv_mean_kl` | Mean full-vs-cache drift. | Lower is better. | Paired logits. |
| `kv_p95_kl` | Tail full-vs-cache drift. | Lower is better. | Paired logits. |
| `kv_mean_js` | Mean JS for full-vs-cache drift. | Lower is better. | Paired logits. |
| `kv_entropy_drift` | Entropy shift introduced by cached decode. | Near zero is best. | Paired logits. |
| `kv_top1_change_rate` | Top-1 instability in cached decode. | Lower is better. | Paired logits. |
| `kv_generation_divergence` | Generated text divergence between full and cached decode. | Lower is better. | Paired deterministic outputs. |
| `device_drift` | Distribution drift between devices or precision variants. | Lower is better. | Paired outputs or logits and device metadata. |

## Interpretation

Serving metrics require paired execution. A batch metric without a single-sample
control is not meaningful. A KV-cache metric without a full-forward control is
not meaningful. Record decoding parameters and deterministic settings.

## Dataset examples

Use a small prompt set replayed across serving modes with identical tokenizer,
model artifact, seed and decoding settings. Keep endpoint URL, hardware class
and precision profile in report metadata, not in committed local notes.

## Action policy

When serving drift appears, first repeat the paired run with deterministic
settings. Then inspect whether drift is restricted to batch, cache, device or
precision mode before treating it as a model-quality regression.
