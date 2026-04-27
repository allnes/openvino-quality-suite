from oviqs.aggregation.gates import evaluate_gates


def test_gates_pass_warning_unknown():
    metrics = {"likelihood": {"nll_delta_vs_ref": 0.2}}
    gates = {"likelihood": {"nll_delta_vs_ref_max": 0.1, "ppl_relative_delta_vs_ref_max": 0.05}}
    result = evaluate_gates(metrics, gates)
    checks = result["sections"]["likelihood"]["checks"]
    assert checks["nll_delta_vs_ref_max"]["status"] == "warning"
    assert checks["nll_delta_vs_ref_max"]["reference_status"] == "present"
    assert checks["ppl_relative_delta_vs_ref_max"]["status"] == "unknown"
    assert result["overall_status"] == "warning"


def test_gates_require_metric_reference():
    metrics = {"custom": {"unreferenced_metric": 1.0}}
    gates = {"custom": {"unreferenced_metric_max": 2.0}}
    result = evaluate_gates(metrics, gates)
    check = result["sections"]["custom"]["checks"]["unreferenced_metric_max"]
    assert check["status"] == "unknown"
    assert check["reference_status"] == "missing"


def test_gates_resolve_nested_metrics_and_references():
    metrics = {
        "serving": {"batch_invariance": {"mean_kl": 0.0}},
        "metric_references": {
            "serving": {
                "batch_invariance": {
                    "primary_reference": "same-sample deterministic serving invariance oracle"
                }
            }
        },
    }
    gates = {"serving": {"batch_invariance_mean_kl_max": 0.005}}
    result = evaluate_gates(metrics, gates)
    check = result["sections"]["serving"]["checks"]["batch_invariance_mean_kl_max"]
    assert check["metric_path"] == "batch_invariance.mean_kl"
    assert check["status"] == "pass"
    assert check["reference_status"] == "present"
