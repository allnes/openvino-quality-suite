from __future__ import annotations

from typing import Any

from oviqs.adapters.integrations.base import (
    IntegrationResult,
    run_callable_integration,
    unavailable_result,
)
from oviqs.adapters.integrations.phoenix_adapter import agent_trace_to_spans
from oviqs.domain.traces import AgentTrace


def ensure_opik_available() -> IntegrationResult:
    try:
        import opik  # noqa: F401
    except ImportError:
        return unavailable_result("opik", "opik", "observability")
    return IntegrationResult(name="opik", status="pass")


def export_to_opik(client, traces: list[AgentTrace], **kwargs) -> IntegrationResult:
    spans: list[dict[str, Any]] = [span for trace in traces for span in agent_trace_to_spans(trace)]
    if hasattr(client, "log_spans"):
        return run_callable_integration("opik", client.log_spans, spans=spans, **kwargs)
    if hasattr(client, "log_traces"):
        return run_callable_integration("opik", client.log_traces, traces=spans, **kwargs)
    if hasattr(client, "track"):
        return run_callable_integration("opik", client.track, traces=spans, **kwargs)
    return IntegrationResult(
        name="opik",
        status="fail",
        error="Opik client must expose log_spans(...), log_traces(...) or track(...).",
    )
