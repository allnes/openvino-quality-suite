from oviqs.domain.metrics.agent import (
    agent_state_drift,
    observation_grounding_score,
    policy_violation_rate,
    recovery_after_tool_error,
    redundant_tool_call_rate,
    task_completion,
    tool_call_validity,
)
from oviqs.domain.traces import AgentTrace, TraceStep


def test_redundant_tool_call_rate():
    trace = AgentTrace(
        id="t1",
        input="x",
        steps=[
            TraceStep(type="tool_call", tool="search", args={"q": "a"}),
            TraceStep(type="tool_call", tool="search", args={"q": "a"}),
        ],
    )
    metrics = redundant_tool_call_rate(trace)
    assert metrics["redundant_tool_call_rate"] == 0.5


def test_agent_state_drift():
    metrics = agent_state_drift({"a": 1, "b": 2}, {"a": 1, "b": 3})
    assert metrics["state_drift_score"] == 0.5
    assert metrics["state_errors"] == {"a": False, "b": True}


def test_tool_call_validity():
    trace = AgentTrace(
        id="t1", input="x", steps=[TraceStep(type="tool_call", tool="send", args={})]
    )
    metrics = tool_call_validity(trace, {"send": {"required": ["to"]}})
    assert metrics["tool_call_validity"] == 0.0


def test_agent_trace_quality_metrics():
    trace = AgentTrace(
        id="t1",
        input="x",
        steps=[
            TraceStep(type="error", content="tool failed"),
            TraceStep(type="tool_call", tool="search", args={"q": "April 16"}),
            TraceStep(type="observation", result="April 16"),
            TraceStep(type="final", content="April 16"),
        ],
    )

    assert observation_grounding_score(trace)["observation_grounding_score"] == 1.0
    assert task_completion(trace)["task_completion"] == 1.0
    assert policy_violation_rate(trace)["policy_violation_rate"] == 0.0
    assert recovery_after_tool_error(trace)["recovery_after_tool_error"] == 1.0
