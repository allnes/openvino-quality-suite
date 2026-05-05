import json

from typer.testing import CliRunner

from oviqs.cli import app


def test_run_gpu_suite_dummy_backend(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "s1",
                        "task_type": "likelihood",
                        "text": "openvino gpu metrics",
                    }
                ),
                json.dumps(
                    {
                        "id": "s2",
                        "task_type": "likelihood",
                        "text": "batch drift metrics",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "gpu_suite.json"

    result = CliRunner().invoke(
        app,
        [
            "run-gpu-suite",
            "--model",
            "dummy",
            "--backend",
            "dummy",
            "--dataset",
            str(dataset),
            "--device",
            "GPU",
            "--window-size",
            "16",
            "--stride",
            "8",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    _assert_likelihood_and_drift(payload)
    _assert_serving_metrics(payload)
    _assert_rag_agent_and_performance(payload)


def _assert_likelihood_and_drift(payload):
    assert payload["likelihood"]["num_tokens"] > 0
    assert payload["inference_equivalence"]["mean_kl"] == 0.0
    assert payload["inference_equivalence"]["top10_overlap"] == 1.0
    assert payload["inference_equivalence"]["top1_changed_rate"] == 0.0


def _assert_serving_metrics(payload):
    assert payload["serving"]["batch_invariance"]["mean_kl"] == 0.0
    assert payload["serving"]["batch_invariance"]["top10_overlap"] == 1.0
    assert payload["serving"]["batch_mean_kl"] == 0.0


def _assert_rag_agent_and_performance(payload):
    assert payload["rag"]["supported_claim_ratio"] == 1.0
    assert payload["rag"]["context_recall"] == 1.0
    assert payload["agent"]["observation_grounding_score"] == 1.0
    assert payload["agent"]["recovery_score"] == 1.0
    assert payload["performance"]["forward_latency_ms_mean"] >= 0.0
