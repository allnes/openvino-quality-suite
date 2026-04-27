from __future__ import annotations

from typing import Any, Protocol

from oviqs.core.trace import AgentTrace, TraceStep


class RedundancyDetector(Protocol):
    def is_redundant(self, call: TraceStep, previous_calls: list[TraceStep]) -> bool: ...


class RuleBasedRedundancyDetector:
    def is_redundant(self, call: TraceStep, previous_calls: list[TraceStep]) -> bool:
        return any(prev.tool == call.tool and prev.args == call.args for prev in previous_calls)


def agent_state_drift(
    expected_state: dict[str, Any], actual_state: dict[str, Any]
) -> dict[str, Any]:
    errors = {key: actual_state.get(key) != value for key, value in expected_state.items()}
    return {
        "state_drift_score": sum(errors.values()) / max(len(errors), 1),
        "state_errors": errors,
    }


def redundant_tool_call_rate(
    trace: AgentTrace, detector: RedundancyDetector | None = None
) -> dict[str, float | int]:
    detector = detector or RuleBasedRedundancyDetector()
    tool_calls = [step for step in trace.steps if step.type == "tool_call"]
    if not tool_calls:
        return {"tool_calls": 0, "redundant_tool_calls": 0, "redundant_tool_call_rate": 0.0}
    redundant = 0
    for i, call in enumerate(tool_calls):
        if detector.is_redundant(call, tool_calls[:i]):
            redundant += 1
    return {
        "tool_calls": len(tool_calls),
        "redundant_tool_calls": redundant,
        "redundant_tool_call_rate": redundant / len(tool_calls),
    }


def tool_call_validity(
    trace: AgentTrace, tool_schemas: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    calls = [step for step in trace.steps if step.type == "tool_call"]
    errors: list[dict[str, Any]] = []
    for idx, call in enumerate(calls):
        if not call.tool or call.tool not in tool_schemas:
            errors.append({"index": idx, "tool": call.tool, "error": "unknown_tool"})
            continue
        schema = tool_schemas[call.tool]
        required = schema.get("required", [])
        for field in required:
            if field not in call.args:
                errors.append(
                    {"index": idx, "tool": call.tool, "field": field, "error": "missing_required"}
                )
        forbidden = schema.get("forbidden", [])
        for field in forbidden:
            if field in call.args:
                errors.append(
                    {"index": idx, "tool": call.tool, "field": field, "error": "forbidden_field"}
                )
    valid = max(len(calls) - len({err["index"] for err in errors}), 0)
    return {
        "tool_calls": len(calls),
        "valid_tool_calls": valid,
        "tool_call_validity": valid / max(len(calls), 1),
        "errors": errors,
    }


def observation_grounding_score_placeholder() -> dict[str, None | list[str]]:
    return {
        "observation_grounding_score": None,
        "warnings": ["observation_grounding_score requires claim extractor or judge_model"],
    }


def observation_grounding_score(trace: AgentTrace) -> dict[str, float | int]:
    """Rule-based grounding: final/tool-call claims should appear in observations."""

    observations = " ".join(
        str(step.result if step.result is not None else step.content or "")
        for step in trace.steps
        if step.type == "observation"
    ).lower()
    claims = [
        step.content.strip()
        for step in trace.steps
        if step.type in {"message", "final"} and step.content and step.content.strip()
    ]
    if not claims:
        return {"claims": 0, "grounded_claims": 0, "observation_grounding_score": 1.0}
    grounded = sum(1 for claim in claims if claim.lower() in observations)
    return {
        "claims": len(claims),
        "grounded_claims": grounded,
        "observation_grounding_score": grounded / len(claims),
    }


def task_completion(trace: AgentTrace) -> dict[str, float | bool]:
    """Check whether a trace reached a final step without a later error."""

    has_final = any(step.type == "final" for step in trace.steps)
    has_error_after_final = False
    seen_final = False
    for step in trace.steps:
        if step.type == "final":
            seen_final = True
        elif seen_final and step.type == "error":
            has_error_after_final = True
    completed = has_final and not has_error_after_final
    return {"task_completed": completed, "task_completion": 1.0 if completed else 0.0}


def policy_violation_rate(trace: AgentTrace) -> dict[str, float | int]:
    """Count trace steps marked with `metadata.policy_violation`."""

    checked_steps = len(trace.steps)
    violations = sum(1 for step in trace.steps if step.metadata.get("policy_violation"))
    return {
        "checked_steps": checked_steps,
        "policy_violations": violations,
        "policy_violation_rate": violations / max(checked_steps, 1),
    }


def recovery_after_tool_error(trace: AgentTrace) -> dict[str, float | int | None]:
    """Measure whether tool errors are followed by another tool call or final answer."""

    error_indices = [idx for idx, step in enumerate(trace.steps) if step.type == "error"]
    if not error_indices:
        return {"tool_errors": 0, "recovered_tool_errors": 0, "recovery_after_tool_error": None}
    recovered = 0
    for idx in error_indices:
        if any(step.type in {"tool_call", "final"} for step in trace.steps[idx + 1 :]):
            recovered += 1
    return {
        "tool_errors": len(error_indices),
        "recovered_tool_errors": recovered,
        "recovery_after_tool_error": recovered / len(error_indices),
    }
