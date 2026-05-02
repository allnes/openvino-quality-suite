from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenerationOutput(BaseModel):
    text: str
    token_ids: list[int] = Field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["GenerationOutput"]
