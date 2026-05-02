from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

TaskType = Literal[
    "likelihood",
    "long_context",
    "controlled_context",
    "rag",
    "agent",
    "generation",
    "serving",
]


class EvalSample(BaseModel):
    id: str
    task_type: TaskType
    text: str | None = None
    prompt: str | None = None
    context: str | None = None
    target: str | None = None
    references: list[str] = Field(default_factory=list)
    retrieved_contexts: list[str] = Field(default_factory=list)
    expected_evidence: list[str] = Field(default_factory=list)
    expected_constraints: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_payload(self) -> EvalSample:
        if self.task_type == "likelihood" and not (self.text or self.prompt):
            raise ValueError("likelihood samples require text or prompt")
        if self.task_type in {"controlled_context", "long_context"} and not self.target:
            raise ValueError("controlled/long-context samples require target")
        return self


class TokenizedSample(BaseModel):
    id: str
    input_ids: list[int]
    attention_mask: list[int] | None = None
    target_mask: list[int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = ["EvalSample", "TaskType", "TokenizedSample"]
