"""Converter tests for the recommended open-dataset loaders.

These exercise the row->sample / row->trace conversion offline (no HuggingFace
download); the load_* functions wrap these converters around datasets.load_dataset.
"""

from __future__ import annotations

from oviqs.adapters.datasets import (
    bfcl_row_to_trace,
    jsonschema_row_to_sample,
    ragas_row_to_sample,
    ruler_row_to_sample,
)


def test_ruler_row_to_sample_maps_long_context_fields():
    sample = ruler_row_to_sample(
        {"index": 3, "input": "Find the needle.", "context": "haystack " * 10, "outputs": ["n42"]},
        task_name="4096",
    )
    assert sample.task_type == "long_context"
    assert sample.prompt == "Find the needle."
    assert sample.target == "n42"
    assert sample.references == ["n42"]
    assert sample.metadata["task"] == "4096"


def test_jsonschema_row_to_sample_keeps_schema_in_constraints():
    sample = jsonschema_row_to_sample(
        {"id": "js1", "json_schema": '{"type": "object", "required": ["status"]}'},
        subset="default",
    )
    assert sample.task_type == "generation"
    assert sample.expected_constraints["json_schema"]["required"] == ["status"]
    assert "schema" in sample.prompt.lower()


def test_ragas_row_to_sample_maps_rag_fields():
    sample = ragas_row_to_sample(
        {
            "id": "r1",
            "question": "Which device runs the check?",
            "contexts": ["OpenVINO runs on Intel GPU.", "Unrelated."],
            "ground_truth": "Intel GPU",
        }
    )
    assert sample.task_type == "rag"
    assert sample.retrieved_contexts == ["OpenVINO runs on Intel GPU.", "Unrelated."]
    assert sample.expected_evidence == ["Intel GPU"]


def test_bfcl_row_to_trace_builds_tool_calls_and_expected_tools():
    trace = bfcl_row_to_trace(
        {
            "id": "b1",
            "question": "What is the weather?",
            "function": [{"name": "get_weather", "parameters": {"required": ["city"]}}],
            "ground_truth": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
        }
    )
    tool_calls = [step for step in trace.steps if step.type == "tool_call"]
    assert tool_calls[0].tool == "get_weather"
    assert tool_calls[0].args == {"city": "Paris"}
    assert trace.expected_tools[0]["name"] == "get_weather"
    assert trace.expected_tools[0]["required"] == ["city"]
    assert any(step.type == "final" for step in trace.steps)
