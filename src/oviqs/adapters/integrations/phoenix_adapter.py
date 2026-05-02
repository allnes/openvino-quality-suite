from __future__ import annotations

from typing import Any

from oviqs.adapters.integrations.base import (
    IntegrationResult,
    run_callable_integration,
    unavailable_result,
)
from oviqs.domain.traces import AgentTrace


def ensure_phoenix_available() -> IntegrationResult:
    try:
        import phoenix  # noqa: F401
    except ImportError:
        return unavailable_result("phoenix", "arize-phoenix", "observability")
    return IntegrationResult(name="phoenix", status="pass")


def agent_trace_to_spans(trace: AgentTrace) -> list[dict[str, Any]]:
    spans = []
    for idx, step in enumerate(trace.steps):
        spans.append(
            {
                "trace_id": trace.id,
                "span_id": f"{trace.id}:{idx}",
                "name": step.tool or step.type,
                "kind": step.type,
                "input": step.args or step.content,
                "output": step.result,
                "timestamp_ms": step.timestamp_ms,
                "metadata": step.metadata,
            }
        )
    return spans


def export_to_phoenix(client, traces: list[AgentTrace], **kwargs) -> IntegrationResult:
    spans = [span for trace in traces for span in agent_trace_to_spans(trace)]
    if hasattr(client, "log_spans"):
        return run_callable_integration("phoenix", client.log_spans, spans=spans, **kwargs)
    if hasattr(client, "log_traces"):
        return run_callable_integration("phoenix", client.log_traces, traces=spans, **kwargs)
    return IntegrationResult(
        name="phoenix",
        status="fail",
        error="Phoenix client must expose log_spans(...) or log_traces(...).",
    )
