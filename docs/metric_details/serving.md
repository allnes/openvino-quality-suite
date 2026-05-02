# Serving Metrics

Serving metrics check whether the same model behaves consistently when the runtime changes
execution mode: single request vs batch, full-forward vs KV-cache decode, or one device vs
another. These metrics are equivalence tests. They should be run with deterministic
settings before application-level quality gates.

Use them to answer:

- Does batching change logits for the same prompt?
- Does cached decode match full-forward logits?
- Does deterministic generation keep the same prefix across serving variants?
- Are OpenVINO stateful model settings, masks or cache positions correct?

## Scientific Basis

For deterministic inference, changing serving mechanics should not materially change the
mathematical next-token distribution for the same model and tokenized input. In practice,
small numeric differences can appear because of precision, kernel selection, padding,
dynamic shapes or cache implementation. OVIQS treats these differences as distribution
drift and top-token stability checks.

For logits `Z_a` from the baseline serving path and `Z_b` from the variant path, OVIQS
reuses distribution drift:

```text
batch_or_cache_drift = drift(softmax(Z_a), softmax(Z_b))
```

For deterministic generated text, prefix divergence checks visible output:

```text
prefix_divergence_rate = 1 - prefix_match_tokens / max(len(a), len(b), 1)
```

## OVIQS Implementation

Source: `src/oviqs/domain/metrics/serving.py`.

Core functions:

- `batch_invariance_drift(alone_logits, batched_logits)`;
- `generation_prefix_divergence(alone_text, batched_text)`;
- `kv_cache_drift(full_logits, cached_logits)`;
- `kv_cache_drift_interface()` placeholder for unsupported runners.

The OpenVINO runtime runner now exposes stateful cached decode when the converted model
supports OpenVINO state.

## Metrics

### batch_invariance_mean_kl

Formula:

```text
mean_t KL(P_alone_t || P_batched_t)
```

Direction: lower is better.

Interpretation:

- Near zero means batching did not change the distribution.
- Non-zero drift can indicate padding, attention-mask, sequence-length, dynamic-shape or
  request-layout issues.
- Compare only the positions belonging to the same prompt; do not score padding tokens.

Action:

- Verify attention masks and position IDs in batched inputs.
- Test prompts with different lengths.
- Compare CPU and GPU to isolate plugin-specific behavior.

### batch_invariance_top1_changed_rate

Formula:

```text
mean_t [argmax(P_alone_t) != argmax(P_batched_t)]
```

Direction: lower is better.

Interpretation:

- Any non-zero value can change deterministic generation.
- If KL is small but top-1 changed, inspect top-token margins.
- If both KL and top-1 change are high, batching is not equivalent and should not be used
  as a release baseline.

### generation_prefix_divergence_prefix_divergence_rate

Formula:

```text
prefix_divergence_rate =
  1 - common_prefix_word_count / max(word_count_a, word_count_b, 1)
```

Direction: lower is better.

Interpretation:

- `0.0` means outputs share the whole shorter/equal prefix under whitespace splitting.
- Non-zero under greedy deterministic decoding means serving behavior changed.
- This is a text-level symptom; use drift metrics to identify whether logits changed.

Action:

- Freeze decoding parameters and random seed.
- Check stop tokens and max output tokens.
- If divergence starts after several tokens, run KV-cache drift.

### kv_cache_drift_mean_kl

Formula:

```text
mean_t KL(P_full_forward_t || P_cached_decode_t)
```

Direction: lower is better.

Interpretation:

- Measures whether stateful one-token decode matches full-forward execution.
- Regressions often point to cache position, past-key-value layout, state reset, input
  shape, attention mask or stateful export problems.
- A model without stateful cache support should report `unknown`, not pass.

Action:

- Confirm the OpenVINO IR exposes state.
- Reset state between samples.
- Check `cache_position`, attention mask and whether cached decode is batch size 1.
- Compare first cached step separately from later steps.

### kv_cache_drift_top1_changed_rate

Formula:

```text
mean_t [argmax(P_full_forward_t) != argmax(P_cached_decode_t)]
```

Direction: lower is better.

Interpretation:

- Non-zero means cached decode can choose different deterministic tokens.
- If only late tokens change, inspect cache growth and position tracking.
- If the first cached token changes, inspect prompt prefill and state initialization.

## Dataset Design

Minimal serving row:

```json
{"id":"s1","task_type":"serving","prompt":"Return JSON with the release date.","max_new_tokens":32}
```

Batch-invariance set:

```json
{"id":"b1","task_type":"serving","prompt":"Short prompt."}
{"id":"b2","task_type":"serving","prompt":"A longer prompt with numbers 16 and 2026."}
```

Recommended datasets:

- two or more prompts with different lengths for batch checks;
- prompts with shared prefixes;
- prompts containing stop tokens, punctuation, code and numerical answers;
- long prompts that force several cached decode steps;
- deterministic generation fixtures with greedy decoding.

Dataset rules:

- Freeze decoding parameters and seed.
- Keep batch composition stable for batch-invariance gates.
- Record device, precision, OpenVINO version, backend and whether the model is stateful.
- Treat unsupported KV-cache checks as `unknown`; do not hide them as pass.

## Decision Guide

| Observation | Likely cause | Action |
|---|---|---|
| Batch KL high | padding/mask/dynamic-shape issue | Inspect batched feeds and compare same prompt alone. |
| Batch top-1 changed | deterministic output may change | Block release until sample-level cause is understood. |
| Prefix divergence only | decoding or stop-token issue | Check generation settings before model export. |
| KV drift high | stateful cache mismatch | Inspect state reset, cache position and IR state exposure. |
| KV unsupported | runner/model lacks state | Mark gate unknown or use a stateful runner. |

## References

- OpenVINO, [Stateful models and State API](https://docs.openvino.ai/2023.3/openvino_docs_OV_UG_stateful_models_intro.html).
- OpenVINO GenAI, [How OpenVINO GenAI Works](https://openvinotoolkit.github.io/openvino.genai/docs/concepts/how-it-works).
- OpenVINO, [PerfMetrics API](https://docs.openvino.ai/2025/api/genai_api/_autosummary/openvino_genai.PerfMetrics.html).
- Vijay Janapa Reddi et al., [MLPerf Inference Benchmark](https://arxiv.org/abs/1911.02549), 2019.
