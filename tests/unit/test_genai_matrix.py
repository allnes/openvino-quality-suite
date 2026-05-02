import json
from pathlib import Path

from typer.testing import CliRunner

from oviqs.application.services.catalogs import load_genai_model_matrix
from oviqs.cli import app
from oviqs.domain.models import export_plan, sanitize_model_name

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = REPO_ROOT / "configs/examples/genai_metric_models.yaml"


def test_load_and_filter_genai_matrix():
    matrix = load_genai_model_matrix(MATRIX_PATH)

    assert matrix.default_model("smoke").id == "Qwen/Qwen2.5-0.5B-Instruct"
    likelihood_models = matrix.list_models(tier="smoke", metric="likelihood")

    assert [model.id for _tier, model in likelihood_models] == [
        "sshleifer/tiny-gpt2",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/Phi-3-mini-4k-instruct",
    ]


def test_export_plan_builds_optimum_commands():
    matrix = load_genai_model_matrix(MATRIX_PATH)
    commands = export_plan(
        matrix,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        output_root="models",
        variants=["eval_logits"],
    )

    assert len(commands) == 1
    assert commands[0].command[:3] == ["optimum-cli", "export", "openvino"]
    assert commands[0].task == "text-generation"
    assert commands[0].output_dir == "models/qwen--qwen2-5-0-5b-instruct-eval_logits"


def test_sanitize_model_name():
    assert sanitize_model_name("Qwen/Qwen2.5-0.5B_Instruct") == "qwen--qwen2-5-0-5b-instruct"


def test_list_genai_models_cli_json():
    result = CliRunner().invoke(
        app,
        [
            "list-genai-models",
            "--config",
            str(MATRIX_PATH),
            "--tier",
            "smoke",
            "--metric",
            "generation",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [item["id"] for item in payload] == [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]


def test_genai_export_plan_cli_json():
    result = CliRunner().invoke(
        app,
        [
            "genai-export-plan",
            "--model",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--config",
            str(MATRIX_PATH),
            "--variant",
            "eval_logits",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["task"] == "text-generation"
