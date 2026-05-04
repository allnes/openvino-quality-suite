# Metric philosophy

OVIQS separates evidence from interpretation. The goal is not to make every run
pass; the goal is to make quality evidence reviewable.

## Principles

- Metric math stays backend-independent where possible.
- References and oracles explain what a metric means.
- Gates decide status from configured thresholds.
- Missing evidence is represented as `unknown`.
- Renderers explain evidence rather than hide it.

This keeps reports useful even when a backend supports only part of the
diagnostic surface.

## Unknown is a real state

`unknown` means the suite does not have enough evidence to make a pass, warning
or fail decision. It can happen because a backend cannot expose logits, a dataset
does not include the required field, a gate path is wrong, or a reference has not
been registered.

Treating `unknown` as pass would make CI look cleaner while hiding risk. OVIQS
keeps the uncertainty visible.

## References and gates

A gateable metric should have reference or oracle metadata. This metadata helps
reviewers understand whether a threshold is based on a known benchmark, a
domain-specific rule or a local policy.

Gates should be narrow. Prefer a small set of reliable metrics over a large set
of ambiguous checks.
