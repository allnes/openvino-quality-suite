from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    type: Literal["message", "tool_call", "observation", "final", "error"]
    role: str | None = None
    content: str | None = None
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    timestamp_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTrace(BaseModel):
    id: str
    input: str
    steps: list[TraceStep]
    expected_tools: list[dict[str, Any]] = Field(default_factory=list)
    expected_state: dict[str, Any] = Field(default_factory=dict)
    expected_constraints: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
