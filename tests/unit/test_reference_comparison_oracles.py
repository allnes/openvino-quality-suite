"""Oracle-status behavior of the reference comparison service.

A non-zero drift (or below-ideal overlap/cosine) is NOT a failure when no gate is
configured: PyTorch-vs-OpenVINO / quantization drift is expected to be non-zero,
so such metrics are reported as `measured`, while the deterministic ideal stays
`pass`. Boolean json/schema oracles keep their pass/fail semantics.
"""

from __future__ import annotations

from oviqs.application.reporting.reference_comparison_service import (
    metric_status,
)


def _report(section: str, **metrics: object) -> dict[str, object]:
    return {"schema_version": "openvino_llm_quality_v1", section: {"status": "pass", **metrics}}


def test_nonzero_drift_is_measured_not_fail():
    report = _report("inference_equivalence", mean_kl=0.28, top1_changed_rate=0.27)
    assert metric_status(report, "inference_equivalence", "mean_kl", 0.28) == "measured"
    assert metric_status(report, "inference_equivalence", "top1_changed_rate", 0.27) == "measured"


def test_zero_drift_is_pass():
    report = _report("serving", kv_mean_kl=0.0)
    assert metric_status(report, "serving", "kv_mean_kl", 9e-6) == "pass"


def test_below_ideal_overlap_is_measured_not_fail():
    report = _report("inference_equivalence", mean_logit_cosine=0.947)
    assert metric_status(report, "inference_equivalence", "mean_logit_cosine", 0.947) == "measured"
    assert metric_status(report, "inference_equivalence", "top5_overlap", 1.0) == "pass"


def test_boolean_json_oracle_keeps_pass_fail():
    report = _report("generation", json_valid=False)
    assert metric_status(report, "generation", "json_valid", False) == "fail"
    report_ok = _report("generation", json_valid=True)
    assert metric_status(report_ok, "generation", "json_valid", True) == "pass"
