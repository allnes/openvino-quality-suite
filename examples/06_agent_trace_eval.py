from oviqs.core.trace import AgentTrace, TraceStep
from oviqs.metrics.agent import redundant_tool_call_rate

trace = AgentTrace(
    id="t1",
    input="search twice",
    steps=[
        TraceStep(type="tool_call", tool="search", args={"q": "x"}),
        TraceStep(type="tool_call", tool="search", args={"q": "x"}),
    ],
)
print(redundant_tool_call_rate(trace))
