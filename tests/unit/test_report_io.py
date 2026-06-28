"""Tests for write_report — used by the remote GPU metric matrix scripts."""

from __future__ import annotations

import json

import numpy as np

from oviqs.domain.reports import EvaluationReport, ReportRun, ReportSummary, write_report


def test_write_report_serializes_plain_dict_with_numpy(tmp_path):
    out = tmp_path / "matrix.json"
    report = {
        "run": {"id": "m", "model": "x"},
        "likelihood": {"nll": np.float32(3.39), "num_tokens": np.int64(996)},
        "inference_equivalence": {"mean_kl": np.float64(0.28), "samples": np.array([1.0, 2.0])},
    }
    write_report(report, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["likelihood"]["num_tokens"] == 996
    assert abs(loaded["likelihood"]["nll"] - 3.39) < 1e-4
    assert loaded["inference_equivalence"]["samples"] == [1.0, 2.0]


def test_write_report_serializes_pydantic_model(tmp_path):
    out = tmp_path / "report.json"
    report = EvaluationReport(
        run=ReportRun(id="r1"),
        summary=ReportSummary(overall_status="pass"),
    )
    write_report(report, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["run"]["id"] == "r1"
    assert loaded["summary"]["overall_status"] == "pass"
