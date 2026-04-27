from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class ModelRunInfo(BaseModel):
    model_id: str
    backend: str
    device: str | None = None
    precision: str | None = None
    model_path: str | None = None
    tokenizer_path: str | None = None
    openvino_version: str | None = None
    openvino_genai_version: str | None = None
    optimum_intel_version: str | None = None
    transformers_version: str | None = None
    tokenizer_hash: str | None = None
    chat_template_hash: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


@dataclass
class LogitsOutput:
    logits: np.ndarray
    input_ids: np.ndarray
    attention_mask: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


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
