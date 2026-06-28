from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from oviqs.adapters.datasets.jsonl import read_jsonl, write_jsonl
from oviqs.adapters.errors import OptionalDependencyError
from oviqs.domain.traces import AgentTrace, TraceStep


def _coerce_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"raw": value}
    return {}


def bfcl_row_to_trace(row: dict[str, Any]) -> AgentTrace:
    """Convert a Berkeley Function Calling Leaderboard row into an AgentTrace.

    Each expected function call becomes a ``tool_call`` step, and the involved
    tool schemas (function name -> required params) are recorded as
    ``expected_tools`` so :func:`tool_call_validity` and tool-correctness metrics
    have ground truth to score against.
    """

    question = row.get("question") or row.get("prompt") or row.get("query") or ""
    if isinstance(question, list):
        question = " ".join(str(part) for part in question)

    functions = row.get("function") or row.get("functions") or row.get("tools") or []
    if isinstance(functions, dict):
        functions = [functions]
    expected_tools: list[dict[str, Any]] = []
    for fn in functions:
        if not isinstance(fn, dict):
            continue
        params = fn.get("parameters") or {}
        required = params.get("required", []) if isinstance(params, dict) else []
        expected_tools.append({"name": fn.get("name"), "required": list(required)})

    calls = row.get("ground_truth") or row.get("answer") or row.get("expected_calls") or []
    if isinstance(calls, dict):
        calls = [calls]
    steps: list[TraceStep] = []
    for call in calls:
        if isinstance(call, dict):
            name = call.get("name") or next(iter(call), None)
            args = call.get("arguments") or call.get("args") or (
                call.get(name) if name in call else {}
            )
            steps.append(TraceStep(type="tool_call", tool=str(name), args=_coerce_args(args)))
        elif isinstance(call, str):
            steps.append(TraceStep(type="tool_call", tool=call, args={}))
    steps.append(TraceStep(type="final", content="done"))

    return AgentTrace(
        id=str(row.get("id") or row.get("_id") or row.get("sample_id")),
        input=str(question),
        steps=steps,
        expected_tools=expected_tools,
        metadata={"source": "bfcl"},
    )


def bfcl_rows_to_traces(rows: list[dict[str, Any]]) -> list[AgentTrace]:
    return [bfcl_row_to_trace(row) for row in rows]


# Recommended open agent / function-calling dataset (Apache-2.0, non-gated).
BFCL_HF_ID = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"


def load_bfcl_traces(
    split: str = "train",
    limit: int = 16,
    cache_path: str | Path | None = None,
) -> list[AgentTrace]:
    """Load Berkeley Function Calling Leaderboard rows into AgentTrace objects.

    Uses the optional ``datasets`` dependency and caches raw rows to
    ``cache_path`` (JSONL) for offline reuse.
    """

    if cache_path is not None and Path(cache_path).exists():
        rows = read_jsonl(cache_path)[:limit]
        return bfcl_rows_to_traces(rows)
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without datasets
        raise OptionalDependencyError("datasets", "agent") from exc
    dataset = load_dataset(BFCL_HF_ID, split=f"{split}[:{limit}]")
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        row = dict(item)
        row.setdefault("id", f"bfcl_{idx:03d}")
        for key in ("function", "functions", "tools", "ground_truth", "answer"):
            if isinstance(row.get(key), (dict, list)):
                row[key] = json.dumps(row[key])
        rows.append(row)
    if cache_path is not None:
        write_jsonl(rows, cache_path)
    # Re-parse JSON-encoded structural fields before converting to traces.
    parsed = read_jsonl(cache_path) if cache_path is not None else rows
    decoded: list[dict[str, Any]] = []
    for row in parsed:
        new_row = dict(row)
        for key in ("function", "functions", "tools", "ground_truth", "answer"):
            if isinstance(new_row.get(key), str):
                with contextlib.suppress(json.JSONDecodeError):
                    new_row[key] = json.loads(new_row[key])
        decoded.append(new_row)
    return bfcl_rows_to_traces(decoded)


__all__ = [
    "BFCL_HF_ID",
    "bfcl_row_to_trace",
    "bfcl_rows_to_traces",
    "load_bfcl_traces",
    "read_jsonl",
]
