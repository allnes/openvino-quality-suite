# Agent metrics

Agent metrics evaluate a structured trace: tool calls, observations, state
updates, final output and recovery after errors. They are workflow metrics, not
free-form answer-quality metrics.

## Metrics

| Metric | Definition | Direction | Required evidence |
|---|---|---|---|
| `tool_call_validity` | Tool calls satisfying schema and policy checks divided by tool calls. | Higher is better. | Tool schemas and trace calls. |
| `tool_correctness` | Expected tools were called correctly. | Higher is better. | Expected tool labels. |
| `argument_correctness` | Tool arguments match expected semantics. | Higher is better. | Expected argument labels or judge. |
| `redundant_tool_call_rate` | Repeated or unnecessary tool calls divided by all calls. | Lower is better. | Ordered tool-call trace. |
| `step_efficiency` | Trace efficiency compared with expected or minimal trajectory. | Higher is better. | Expected trajectory or judge. |
| `agent_state_drift` | Mismatch between expected and actual state. | Lower is better. | Expected and actual state dictionaries. |
| `state_drift_score` | Wrong state fields divided by expected fields. | Lower is better. | Expected and actual state dictionaries. |
| `observation_grounding_score` | Final claims grounded in tool observations. | Higher is better. | Observation steps and final claims. |
| `task_completion` | Trace reached expected final outcome. | Higher or `true` is better. | Final step and expected outcome. |
| `policy_violation_rate` | Steps marked as policy violations divided by checked steps. | Lower is better. | Step metadata. |
| `recovery_score` | Tool-error scenarios recovered successfully. | Higher is better. | Error and recovery labels. |
| `recovery_after_tool_error` | Tool errors followed by recovery action. | Higher is better. | Ordered error/tool/final steps. |
| `same_error_repeat_rate` | Same tool error repeated. | Lower is better. | Error signatures. |
| `fallback_quality_score` | Fallback behavior quality after errors. | Higher is better. | Fallback rubric or labels. |
| `unsafe_recovery_rate` | Unsafe recovery behavior after errors. | Lower is better. | Safety labels. |
| `unnecessary_user_clarification_rate` | Avoidable clarification turns. | Lower is better. | Expected action labels. |

## Interpretation

Agent metrics depend on trace completeness. A score without tool schemas,
expected state or expected outcomes is usually not gateable. Keep rule-based
trace metrics separate from external judge metrics.

## Dataset examples

Use JSONL traces with stable step ids, step type, tool name, arguments,
observation payload, final output and metadata. A paired expected file should
record required tools, forbidden tools, expected state and the target outcome.

## Action policy

When an agent metric fails, inspect the exact trace segment before changing
thresholds. For recovery failures, start at the first `error` step and verify
whether a later tool call or final response actually corrected the problem.
