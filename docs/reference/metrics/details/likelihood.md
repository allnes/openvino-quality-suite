# Likelihood metrics

Likelihood metrics measure how much probability a causal language model assigns
to target tokens under teacher forcing. They are the preferred regression signal
when the backend can expose token-aligned logits or log probabilities.

## Metrics

| Metric | Definition | Direction | Required evidence |
|---|---|---|---|
| `token_logprobs` | Per-token log probability for `x_t` conditioned on previous tokens after the causal one-token shift. | Higher is better per token. | Logits or log probabilities, `input_ids`, attention mask and score mask. |
| `mean_log_prob` | Mean of scored `token_logprobs`. It is `-nll`. | Higher is better. | Same as `token_logprobs`. |
| `num_tokens` | Count of tokens included in the denominator. | Context only. | Score mask after padding and target filtering. |
| `nll` | Negative mean conditional log probability over selected tokens. | Lower is better. | Token-aligned log probabilities. |
| `nll_delta_vs_ref` | Current NLL minus reference NLL on the same samples. | Lower is better; zero means no drift. | Current and reference reports over the same dataset. |
| `perplexity` | `exp(nll)`. Aggregate by averaging NLL first. | Lower is better. | Same as `nll`. |
| `ppl_relative_delta_vs_ref` | `(current_ppl - reference_ppl) / reference_ppl`. | Lower is better. | Current and reference PPL on the same tokenization. |
| `sliding_window_ppl` | PPL over overlapping windows without double-counting context tokens. | Lower is better. | Runner capable of repeated forward passes, window size and stride. |
| `word_perplexity` | PPL normalized to a word denominator for tokenizer-sensitive comparisons. | Lower is better within the declared normalization. | Word counts and token-level NLL. |
| `byte_perplexity` | PPL normalized to byte count. | Lower is better within the declared normalization. | UTF-8 byte counts and token-level NLL. |
| `bits_per_byte` | Byte-normalized negative log-likelihood in bits. | Lower is better. | Byte counts and token-level NLL. |
| `length_bucketed_ppl` | PPL grouped by sample length buckets. | Lower is better per bucket. | Sample token counts and per-token NLL. |
| `effective_context_bucketed_ppl` | PPL grouped by available left-context buckets. | Lower is better per bucket. | Absolute token positions and context window metadata. |
| `position_bucketed_ppl` | PPL grouped by relative prompt position. | Lower is better per bucket. | Per-token NLL and sequence length. |

## Interpretation

PPL is exponential in NLL, so small NLL changes can look large after
exponentiation. Review `nll`, `num_tokens` and tokenization before making a
release decision from PPL alone. Compare PPL only across the same tokenizer,
dataset slice, target mask and context policy.

## Dataset examples

Use WikiText-2 or another fixed public corpus for broad language-model smoke
tests. Use deterministic JSONL fixtures in CI when exact regression behavior is
more important than benchmark coverage. Each row should have a stable sample id
and either prompt text with recorded tokenizer identity or pre-tokenized ids.

## Action policy

Gate `nll` or `perplexity` only after the dataset, tokenizer, score mask and
context length policy are documented. If a backend cannot return aligned
likelihood evidence, mark these metrics `unknown` and switch to generation or
serving metrics that match the evidence actually collected.
