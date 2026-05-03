from pathlib import Path

import pytest

from oviqs.adapters.reporting import CanonicalReportWriter
from oviqs.domain.reports import EvaluationReport, ReportRun
from oviqs.domain.samples import EvalSample, TokenizedSample
from oviqs.domain.traces import AgentTrace, TraceStep


def test_eval_sample_schema():
    sample = EvalSample(id="s1", task_type="likelihood", text="hello")
    assert sample.id == "s1"
    with pytest.raises(ValueError):
        EvalSample(id="bad", task_type="likelihood")


def test_tokenized_and_trace_schema():
    tokenized = TokenizedSample(id="s1", input_ids=[1, 2], target_mask=[0, 1])
    trace = AgentTrace(id="t1", input="x", steps=[TraceStep(type="final", content="done")])
    assert tokenized.target_mask == [0, 1]
    assert trace.steps[0].type == "final"


def test_report_writer(tmp_path: Path):
    report = EvaluationReport(run=ReportRun(id="run1"), likelihood={"ppl": 2.0})
    out = tmp_path / "report.json"
    CanonicalReportWriter().write(report, out)
    assert '"ppl": 2.0' in out.read_text(encoding="utf-8")
