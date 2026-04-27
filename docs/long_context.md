# Long Context

Controlled tasks should vary context length, fact position and noise mode. OVIQS reports
position-bucketed PPL and lost-in-the-middle scores so long-context failures are localized.

## Recommended axes

- Context length: short baseline, medium window, target production length and stress length.
- Position: beginning, first quartile, middle, third quartile and end.
- Noise: no distractor, irrelevant distractor, conflicting fact and dense retrieval context.
- Target type: exact fact, paraphrase, synthesis, abstention and citation-backed answer.

## Metrics

- `context_gain`: improvement from additional context.
- `degradation_slope`: quality trend as length increases.
- `lost_in_middle_score`: middle-position degradation relative to edge positions.
- `distractor_sensitivity`: degradation from irrelevant or conflicting context.
- Position-bucketed PPL: token-level likelihood localized by context region.

Use `oviq metric-long-context` for small precomputed position-bucket checks:

```bash
oviq metric-long-context --position-ppl-json '{"begin": 3.1, "middle": 4.8, "end": 3.3}'
```
