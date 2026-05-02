from oviqs.domain.metrics.agent import redundant_tool_call_rate
from oviqs.domain.traces import AgentTrace, TraceStep

trace = AgentTrace(
    id="t1",
    input="search twice",
    steps=[
        TraceStep(type="tool_call", tool="search", args={"q": "x"}),
        TraceStep(type="tool_call", tool="search", args={"q": "x"}),
    ],
)
print(redundant_tool_call_rate(trace))
