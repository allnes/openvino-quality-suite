# RAG And Agent

RAG metrics are separated into retrieval, packing, generation and citation quality.
Agent metrics cover tool validity, redundant calls, state drift and observation grounding.
Judge-based scores are opt-in and must not be faked when unavailable.

## RAG sections

- Retrieval: hit rate, expected evidence coverage and distractor rate.
- Packing: context truncation, ordering, duplication and citation availability.
- Generation: answerability, contradiction rate and abstention behavior.
- Citation: supported claim ratio and unsupported citation detection.

## Agent trace sections

Agent traces use ordered steps with `message`, `tool_call`, `observation`, `final` and
`error` events. Metrics should distinguish:

- Tool call validity.
- Redundant tool call rate.
- Observation grounding.
- State drift.
- Recovery after tool or environment errors.
- Policy or constraint violations from `expected_constraints`.

Judge-backed values should remain `None` or omitted unless the configured evaluator
actually ran.
