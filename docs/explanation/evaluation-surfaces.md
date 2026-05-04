# Evaluation surfaces

OVIQS separates diagnostics by the evidence a backend can expose. This keeps the
report honest: if the backend cannot provide the evidence required for a metric,
the metric remains `unknown`.

## Logits-level diagnostics

Logits-level diagnostics include likelihood, distribution drift and
long-context perplexity. They require aligned token positions and access to the
full next-token distribution.

Use them for:

- negative log likelihood and perplexity;
- KL, JS and entropy drift;
- logit cosine similarity;
- top-k overlap;
- token-rank movement.

These metrics are the right choice for runtime equivalence, export validation
and model conversion checks.

## Generation-level diagnostics

Generation-level diagnostics include output structure, RAG, agent traces and
serving checks. They should not be used to fabricate full-distribution metrics.

Use them for:

- output structure and repetition;
- RAG citation and faithfulness signals;
- agent trace grounding and tool-use checks;
- serving invariance and prefix divergence.

Generation-level checks can be useful even when logits are unavailable, but they
must not be presented as substitutes for full distribution metrics.

## Status policy

Missing evidence should produce `unknown`. A backend limitation is different
from a passing quality check, and the report should preserve that distinction.
