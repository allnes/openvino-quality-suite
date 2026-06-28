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

    assert matrix.default_model("target_gpu").id == "Qwen/Qwen3-0.6B"
    likelihood_models = matrix.list_models(tier="target_gpu", metric="likelihood")

    assert [model.id for _tier, model in likelihood_models] == [
        "Qwen/Qwen3-0.6B",
        "google/gemma-2-9b-it",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "openai/gpt-oss-20b",
        "microsoft/phi-4",
    ]


def test_export_plan_builds_optimum_commands():
    matrix = load_genai_model_matrix(MATRIX_PATH)
    commands = export_plan(
        matrix,
        model_id="Qwen/Qwen3-0.6B",
        output_root="models",
        variants=["eval_logits"],
    )

    assert len(commands) == 1
    assert commands[0].command[:3] == ["optimum-cli", "export", "openvino"]
    assert commands[0].task == "text-generation"
    assert commands[0].output_dir == "models/qwen--qwen3-0-6b-eval_logits"


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
            "target_gpu",
            "--metric",
            "generation",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [item["id"] for item in payload] == [
        "Qwen/Qwen3-0.6B",
        "google/gemma-2-9b-it",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "openai/gpt-oss-20b",
        "microsoft/phi-4",
    ]


def test_genai_export_plan_cli_json():
    result = CliRunner().invoke(
        app,
        [
            "genai-export-plan",
            "--model",
            "Qwen/Qwen3-0.6B",
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
