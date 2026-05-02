# Likelihood Metrics

Likelihood metrics evaluate the probability that a causal language model assigns to known
target tokens. In OVIQS they are the first quality check for export, quantization, device
and runtime changes because they inspect the model distribution before decoding strategy,
retrieval, tools or prompt formatting add extra noise.

Use them to answer:

- Did the OpenVINO model keep the same token probabilities as the reference model?
- Did quantization, device change or graph conversion increase language-model loss?
- Is a long-context regression caused by probability degradation or by downstream logic?

## Scientific Basis

For a token sequence `x_1 ... x_T`, a causal LM factorizes the sequence probability as:

```text
P(x_1 ... x_T) = product_t P(x_t | x_<t)
```

OVIQS scores the shifted targets: logits at position `t - 1` predict token `x_t`.
For selected tokens `S`, the corpus negative log likelihood is:

```text
NLL = - (1 / |S|) * sum_{t in S} log P(x_t | x_<t)
```

Perplexity is the exponentiated mean NLL:

```text
PPL = exp(NLL)
```

The logarithm is natural log, so NLL is measured in nats. If a report needs bits per
token, divide NLL by `ln(2)`.

Important aggregation rule: average token NLL first, then exponentiate. Averaging
per-sample or per-token perplexities gives a biased number because `exp` is nonlinear.

## OVIQS Implementation

Source: `src/oviqs/domain/metrics/likelihood.py`.

Required inputs:

- `logits` with shape `[batch, tokens, vocab]`;
- `input_ids` with the same `[batch, tokens]` prefix dimensions;
- optional `attention_mask`, shifted by one token before scoring;
- optional `target_mask`, also shifted, for answer-only or span-only scoring.

OVIQS converts logits to `float32`, applies `log_softmax`, gathers the log probability of
each target token, applies masks, and aggregates only selected target positions.

## Metrics

### mean_log_prob

Formula:

```text
mean_log_prob = (1 / |S|) * sum_{t in S} log P(x_t | x_<t)
```

Direction: higher is better.

Interpretation:

- Near `0` means the model assigns very high probability to selected targets.
- More negative values mean lower confidence.
- A drop with stable `num_tokens` usually indicates tokenization, logits alignment,
  precision, quantization or model export drift.

What to do:

- Check that tokenizer files are identical between reference and current models.
- Check the one-token causal shift: `logits[:, :-1]` must score `input_ids[:, 1:]`.
- Check masks when scoring answer-only datasets; an off-by-one mask can hide or create a
  regression.

### nll

Formula:

```text
nll = -mean_log_prob
```

Direction: lower is better.

Interpretation:

- Stable NLL across CPU/GPU or FP16/INT8 means the target-token distribution is stable.
- Increased NLL means the current model assigns less probability to known-good text.
- Use NLL for gates instead of PPL when differences are small; NLL is additive and easier
  to aggregate.

Example gate:

```yaml
likelihood:
  nll_max: 2.85
```

### perplexity

Formula:

```text
perplexity = exp(nll)
```

Direction: lower is better.

Interpretation:

- PPL can be read as the model's effective average branching factor under the evaluated
  distribution.
- PPL is easy to communicate, but differences grow exponentially with NLL.
- Use the same tokenizer, corpus normalization and context length whenever comparing PPL.

Example:

```text
nll = 2.30
ppl = exp(2.30) = 9.97
```

### num_tokens

Formula:

```text
num_tokens = |S|
```

Direction: higher is usually more stable; it is not a quality metric by itself.

Interpretation:

- Very low token counts make NLL/PPL noisy.
- A sudden count change often means a tokenizer, attention mask, target mask or dataset
  preprocessing change.
- Do not compare PPL from two reports unless `num_tokens` and dataset selection are
  compatible.

### sliding_window_ppl

Sliding-window PPL evaluates sequences longer than the model's practical context window.
OVIQS runs overlapping windows and scores only the newly introduced stride region, so
overlapping tokens are not double-counted.

Parameters:

- `window_size`: total tokens visible to each forward pass;
- `stride`: number of new token positions scored per window.

Interpretation:

- Short-context PPL stable but sliding-window PPL worse: inspect position IDs, attention
  masks, max context, dynamic shapes and truncation.
- Larger `window_size` should generally improve or saturate PPL if useful context is
  available.
- A very small `stride` is slower but gives more context to each scored token.

### position_bucketed_ppl

Position-bucketed PPL groups token losses by relative position, for example `0_10`,
`30_50`, `50_70`, `90_100`.

Interpretation:

- Edge buckets good and middle buckets worse can indicate lost-in-the-middle behavior.
- End bucket degradation can indicate context-limit, RoPE scaling, attention-mask or
  cache-position issues.
- Use this together with long-context metrics before changing retrieval ordering.

## Dataset Design

Minimal JSONL:

```json
{"id":"lm1","task_type":"likelihood","text":"OpenVINO runs language model inference on Intel GPU."}
```

Answer-only scoring with a target mask:

```json
{
  "id": "qa1",
  "task_type": "likelihood",
  "text": "Question: What device runs the model?\nAnswer: Intel GPU",
  "target_span": "Intel GPU"
}
```

Recommended datasets:

- deterministic smoke JSONL for CI and fast regression checks;
- WikiText-2 validation for quick language-model signal;
- task-specific production prompts for release gates;
- long documents with stable preprocessing for sliding-window PPL;
- lm-evaluation-harness or LightEval corpora for external comparability.

Dataset rules:

- Freeze tokenizer, normalization, prompt template and truncation policy.
- Keep enough tokens per bucket; small buckets make position metrics unstable.
- Separate prompt-only, answer-only and full-text scoring because they answer different
  questions.
- Record model artifact, device, precision, OpenVINO version and backend.

## Decision Guide

| Observation | Likely cause | Action |
|---|---|---|
| `num_tokens` changed | tokenizer, masks, truncation or dataset changed | Fix data path before interpreting NLL. |
| NLL and PPL regress together | probability distribution changed | Compare logits with distribution drift metrics. |
| NLL stable, generation changed | decoding or serving path changed | Check generation and serving metrics. |
| Sliding-window PPL regresses only at long lengths | context handling issue | Inspect position IDs, attention masks, max sequence length and dynamic shapes. |
| Answer-only NLL regresses but prompt NLL is stable | task answer behavior changed | Inspect target span labels and task-specific prompts. |

## References

- Claude Shannon, [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf), Bell System Technical Journal, 1948.
- Hugging Face Transformers, [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/v4.32.0/perplexity).
- Hugging Face LightEval, [Metric list](https://huggingface.co/docs/lighteval/v0.11.0/en/metric-list).
- Stephen Merity et al., [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843), introducing WikiText.
