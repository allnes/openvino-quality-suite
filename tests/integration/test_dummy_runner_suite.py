import json

from typer.testing import CliRunner

from oviqs.cli import app


def test_dummy_likelihood_cli(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps({"id": "s1", "task_type": "likelihood", "text": "a b c"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    result = CliRunner().invoke(
        app,
        [
            "eval-likelihood",
            "--model",
            "dummy",
            "--backend",
            "dummy",
            "--dataset",
            str(dataset),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["likelihood"]["num_tokens"] > 0
