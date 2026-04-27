# Agent Metrics

Agent metrics evaluate ordered traces rather than a single text answer. They measure tool
use, state transitions, observation grounding, completion, policy flags and recovery after
tool errors. These metrics are intentionally deterministic where possible because agent
systems already contain many moving parts.

Use them to answer:

- Did the agent call allowed tools with required arguments?
- Is the agent repeating identical tool calls?
- Did expected workflow state match actual state?
- Is the final answer grounded in observations?
- Did the agent recover after tool errors?

## Scientific Basis

Agent evaluation extends language-model evaluation into an interactive setting where the
model alternates between reasoning, actions/tools, observations and final answers. ReAct
popularized interleaving reasoning and acting; AgentBench evaluates agents across multiple
interactive environments. OVIQS focuses on trace-level invariants that can be checked
without relying on a judge model.

Trace-level evaluation is useful because many production failures are not captured by
answer accuracy alone:

- a correct answer may be produced after an unsafe or invalid tool call;
- a tool loop may waste latency/cost before eventually finishing;
- an ungrounded final answer may ignore the observation that should support it;
- failed recovery may be hidden if only successful examples are tested.

## OVIQS Implementation

Source: `src/oviqs/metrics/agent.py`.

Required inputs:

- `AgentTrace` with ordered steps such as `message`, `tool_call`, `observation`, `final`
  and `error`;
- expected rows with `tool_schemas`, `expected_state` and `actual_state`;
- optional `metadata.policy_violation` flags on trace steps.

Core functions:

- `tool_call_validity(trace, tool_schemas)`;
- `redundant_tool_call_rate(trace, detector=None)`;
- `agent_state_drift(expected_state, actual_state)`;
- `observation_grounding_score(trace)`;
- `task_completion(trace)`;
- `policy_violation_rate(trace)`;
- `recovery_after_tool_error(trace)`.

## Metrics

### tool_call_validity

Formula:

```text
tool_call_validity = valid_tool_calls / total_tool_calls
```

A call is invalid when:

- the tool name is missing or unknown;
- a required argument is missing;
- a forbidden field is present.

Direction: higher is better.

Interpretation:

- Low validity means tool schema alignment broke.
- A single call can produce multiple validation errors, but validity counts whether a call
  had at least one error.
- This does not check semantic correctness of argument values unless encoded in schemas or
  external validators.

Action:

- Version tool schemas with the dataset.
- Add required fields for production-critical arguments.
- Add forbidden fields for internal-only or unsafe parameters.

### redundant_tool_call_rate

Formula:

```text
redundant_tool_call_rate =
  redundant_tool_calls / total_tool_calls
```

Default detector: a call is redundant when a previous call used the same tool name and the
same arguments.

Direction: lower is better.

Interpretation:

- High values indicate loops, memory failure, planner uncertainty or poor observation use.
- Some repeated calls are legitimate if state changes externally; use a custom detector for
  those workflows.

Action:

- Inspect repeated call sequences and observations between them.
- Add loop guards or cache previous tool results.
- Improve prompt instructions for when to stop searching.

### agent_state_drift

Formula:

```text
state_drift_score =
  mismatched_expected_state_keys / expected_state_keys
```

Direction: lower is better.

Interpretation:

- `0.0` means every expected state key matches.
- Missing actual keys count as mismatches.
- Only keys present in `expected_state` are checked.

Action:

- Use for workflows with explicit state machines: ticket triage, retrieval steps, approval
  flows, tool orchestration.
- Keep expected state small and meaningful; do not gate incidental metadata.

### observation_grounding_score

Formula:

```text
observation_grounding_score =
  grounded_message_or_final_claims / message_or_final_claims
```

Current implementation treats each message/final content string as a claim and counts it
as grounded when it appears literally in observations.

Direction: higher is better.

Interpretation:

- Conservative exact-match grounding.
- Works well for short factual outputs, IDs, dates, statuses and tool-returned strings.
- Too strict for paraphrases; use a claim extractor or judge-backed scorer for natural
  language summaries.

Action:

- Normalize observations if tools return structured JSON.
- Add claim extraction for multi-sentence final answers.
- Review low grounding before accepting agent output even if task completion is high.

### task_completion

Formula:

```text
task_completion = 1.0 if trace has final step and no later error else 0.0
```

Direction: higher is better.

Interpretation:

- A final answer before a later error is not considered successful.
- Completion does not measure answer correctness; combine with grounding and task-specific
  labels.

Action:

- If completion drops, inspect orchestration, timeout, tool availability and recovery
  logic.

### policy_violation_rate

Formula:

```text
policy_violation_rate =
  steps_with_metadata_policy_violation / total_steps
```

Direction: lower is better.

Interpretation:

- OVIQS relies on trace metadata; it does not infer policy violations by itself.
- Use this for workflow-specific constraints such as "do not call write tool before
  approval" or "do not expose internal-only fields."

Action:

- Define policy flags close to the instrumentation layer.
- Keep violation categories in metadata for sample-level analysis.

### recovery_after_tool_error

Formula:

```text
recovery_after_tool_error =
  tool_errors_followed_by_tool_call_or_final / tool_errors
```

Direction: higher is better.

If a trace has no error steps, the metric returns `null` because no recovery scenario was
tested.

Interpretation:

- `null` should not be treated as pass for recovery gates.
- Low recovery means the agent stalls or terminates incorrectly after tool failure.
- High recovery does not guarantee final correctness; combine with completion and
  grounding.

Action:

- Add explicit failure fixtures for every critical tool.
- Test transient errors, validation errors and empty results separately.

## Dataset Design

Trace row:

```json
{
  "id": "a1",
  "input": "Find the release date.",
  "steps": [
    {"type":"tool_call","tool":"search","args":{"query":"release date"}},
    {"type":"observation","result":"Release date: April 16, 2026"},
    {"type":"final","content":"April 16, 2026"}
  ]
}
```

Expected row:

```json
{
  "id": "a1",
  "tool_schemas": {"search":{"required":["query"],"forbidden":["internal_field"]}},
  "expected_state": {"done": true},
  "actual_state": {"done": true}
}
```

Failure-recovery row:

```json
{
  "id": "a2",
  "input": "Retry after search timeout.",
  "steps": [
    {"type":"tool_call","tool":"search","args":{"query":"release date"}},
    {"type":"error","content":"timeout"},
    {"type":"tool_call","tool":"search","args":{"query":"release date site:docs"}},
    {"type":"observation","result":"April 16, 2026"},
    {"type":"final","content":"April 16, 2026"}
  ]
}
```

Recommended datasets:

- deterministic trace fixtures with versioned tool schemas;
- failure-recovery traces with explicit `error` steps;
- production traces normalized into `AgentTrace`;
- AgentBench-style environment tasks for broader evaluation;
- DeepEval/OpenEvals/agentevals datasets for judge-backed tool and task correctness.

Dataset rules:

- Store traces in chronological order.
- Keep tool schemas with the evaluation set.
- Include both success and failure paths.
- Separate workflow-state correctness from final-answer correctness.
- Do not gate recovery unless the dataset contains tool errors.

## Decision Guide

| Observation | Likely cause | Action |
|---|---|---|
| Tool validity low | schema/prompt mismatch | Update schemas, tool adapter or tool-call prompt. |
| Redundancy high | loop or memory failure | Add loop guard, result cache or stop criteria. |
| State drift high | workflow state mismatch | Inspect orchestration transitions and state serialization. |
| Grounding low | answer ignores observations | Improve observation use or add claim-level judge. |
| Completion low | orchestration/timeout issue | Inspect trace endings and error handling. |
| Recovery `null` | no error scenario tested | Add failure fixtures before gating recovery. |

## References

- Shunyu Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629), ICLR 2023.
- Xiao Liu et al., [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688), ICLR 2024.
- DeepEval, [Tool Correctness Metric](https://deepeval.com/docs/metrics-tool-correctness).
- DeepEval, [Task Completion Metric](https://deepeval.com/docs/metrics-task-completion).
