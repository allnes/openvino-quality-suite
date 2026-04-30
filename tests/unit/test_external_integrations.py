import json

from oviqs.core.sample import EvalSample
from oviqs.core.trace import AgentTrace, TraceStep
from oviqs.datasets.helmet import helmet_row_to_sample
from oviqs.datasets.longbench import longbench_row_to_sample
from oviqs.datasets.ruler import ruler_row_to_sample
from oviqs.integrations.base import run_callable_integration
from oviqs.integrations.lighteval_adapter import LightEvalTask, evaluate_with_lighteval
from oviqs.integrations.openevals_adapter import evaluate_with_openevals
from oviqs.integrations.phoenix_adapter import agent_trace_to_spans
from oviqs.integrations.promptfoo_adapter import import_promptfoo_results
from oviqs.integrations.ragas_adapter import build_ragas_rows


def test_callable_integration_normalizes_dict():
    result = run_callable_integration("x", lambda value: {"score": value}, 1.0)
    assert result.status == "pass"
    assert result.metrics["score"] == 1.0


def test_ragas_row_builder_uses_answers():
    sample = EvalSample(
        id="r1",
        task_type="rag",
        prompt="question",
        retrieved_contexts=["ctx"],
        expected_evidence=["ctx"],
    )
    rows = build_ragas_rows([sample], [{"id": "r1", "answer": "answer", "reference": "ref"}])
    assert rows == [
        {
            "id": "r1",
            "user_input": "question",
            "response": "answer",
            "retrieved_contexts": ["ctx"],
            "reference": "ref",
            "reference_contexts": ["ctx"],
        }
    ]


def test_lighteval_adapter_uses_python_pipeline():
    class Pipeline:
        def evaluate(self, tasks):
            return {"tasks": tasks}

    result = evaluate_with_lighteval(
        Pipeline(),
        [LightEvalTask("gsm8k", suite="custom", fewshot=5)],
    )
    assert result.status == "pass"
    assert result.metrics["tasks"] == ["custom|gsm8k|5"]


def test_openevals_adapter_uses_callable_evaluator():
    result = evaluate_with_openevals(
        lambda inputs, outputs, reference_outputs: {"ok": outputs == reference_outputs},
        inputs={"q": "x"},
        outputs="a",
        reference_outputs="a",
    )
    assert result.metrics["ok"] is True


def test_promptfoo_result_import(tmp_path):
    path = tmp_path / "promptfoo.json"
    path.write_text(json.dumps({"results": [{"score": 1.0}]}), encoding="utf-8")
    result = import_promptfoo_results(path)
    assert result.status == "pass"
    assert result.metrics["results"][0]["score"] == 1.0


def test_agent_trace_to_observability_spans():
    trace = AgentTrace(
        id="t1",
        input="x",
        steps=[TraceStep(type="tool_call", tool="search", args={"q": "x"}, result="y")],
    )
    spans = agent_trace_to_spans(trace)
    assert spans[0]["trace_id"] == "t1"
    assert spans[0]["name"] == "search"


def test_external_dataset_converters():
    assert (
        longbench_row_to_sample({"id": "1", "input": "q", "context": "c", "answer": "a"}).target
        == "a"
    )
    assert (
        helmet_row_to_sample({"id": "2", "prompt": "q", "documents": ["a"], "answer": "b"}).context
        == "a"
    )
    assert ruler_row_to_sample({"id": "3", "input": "q", "outputs": ["needle"]}).target == "needle"
