import json

from typer.testing import CliRunner

from oviqs.cli import app


def _report_payload(run_id: str, **sections):
    return {
        "schema_version": "openvino_llm_quality_v1",
        "run": {
            "id": run_id,
            "suite": "integration",
            "created_at": "2026-05-03T00:00:00Z",
        },
        "summary": {"overall_status": "pass"},
        **sections,
    }


def test_eval_long_context_cli_computes_metrics(tmp_path):
    dataset = tmp_path / "long.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "l1",
                "task_type": "long_context",
                "context": "alpha beta gamma",
                "target": "delta",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "long_report.json"

    result = CliRunner().invoke(
        app,
        [
            "eval-long-context",
            "--dataset",
            str(dataset),
            "--out",
            str(out),
            "--model",
            "dummy",
            "--backend",
            "dummy",
            "--lengths",
            "12,16",
            "--window-size",
            "8",
            "--stride",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["long_context"]["context_gain"]
    assert "lost_in_middle_score" in payload["long_context"]


def test_eval_serving_cli_computes_batch_metrics(tmp_path):
    out = tmp_path / "serving.json"
    result = CliRunner().invoke(
        app,
        ["eval-serving", "--out", str(out), "--model", "dummy", "--backend", "dummy"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["serving"]["batch_invariance"]["mean_kl"] == 0.0
    assert payload["serving"]["generation_prefix_divergence"]["prefix_divergence_rate"] == 0.0
    assert "batch_invariance" in payload["metric_references"]["serving"]


def test_eval_rag_cli_computes_rule_based_metrics(tmp_path):
    dataset = tmp_path / "rag.jsonl"
    answers = tmp_path / "answers.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "r1",
                "task_type": "rag",
                "retrieved_contexts": ["Doc A says April 16.", "Noise"],
                "expected_evidence": ["April 16"],
                "references": ["doc-a"],
                "metadata": {"relevant_context_indices": [0]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    answers.write_text(
        json.dumps(
            {
                "id": "r1",
                "answer": "April 16",
                "claims": ["April 16"],
                "citations": ["doc-a"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "rag_report.json"

    result = CliRunner().invoke(
        app,
        ["eval-rag", "--dataset", str(dataset), "--answers", str(answers), "--out", str(out)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["rag"]["evidence_coverage"] == 1.0
    assert payload["rag"]["citation_recall"] == 1.0


def test_eval_agent_cli_computes_trace_metrics(tmp_path):
    traces = tmp_path / "traces.jsonl"
    expected = tmp_path / "expected.jsonl"
    traces.write_text(
        json.dumps(
            {
                "id": "a1",
                "input": "find date",
                "steps": [
                    {"type": "tool_call", "tool": "search", "args": {"query": "date"}},
                    {"type": "observation", "result": "April 16"},
                    {"type": "final", "content": "April 16"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    expected.write_text(
        json.dumps(
            {
                "id": "a1",
                "tool_schemas": {"search": {"required": ["query"]}},
                "expected_state": {"done": True},
                "actual_state": {"done": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "agent_report.json"

    result = CliRunner().invoke(
        app,
        ["eval-agent", "--traces", str(traces), "--expected", str(expected), "--out", str(out)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["agent"]["tool_call_validity"] == 1.0
    assert payload["agent"]["task_completion"] == 1.0


def test_list_metric_references_cli():
    result = CliRunner().invoke(app, ["list-metric-references", "--family", "rag", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["family"] == "rag"
    assert "context_precision" in payload[0]["metric_names"]


def test_reference_comparison_cli_writes_metric_reference_table(tmp_path):
    qwen = tmp_path / "qwen.json"
    mistral = tmp_path / "mistral.json"
    qwen.write_text(
        json.dumps(
            _report_payload(
                "qwen",
                likelihood={"status": "pass", "perplexity": 29.6},
                serving={"status": "pass", "batch_mean_kl": 0.0},
                metric_references={},
            )
        ),
        encoding="utf-8",
    )
    mistral.write_text(
        json.dumps(
            _report_payload(
                "mistral",
                likelihood={"status": "pass", "perplexity": 7.8},
                serving={"status": "pass", "batch_mean_kl": 0.0},
                metric_references={},
            )
        ),
        encoding="utf-8",
    )
    out = tmp_path / "comparison.md"

    result = CliRunner().invoke(
        app,
        [
            "report",
            "reference-comparison",
            "--report",
            f"Qwen={qwen}",
            "--report",
            f"Mistral={mistral}",
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    content = out.read_text(encoding="utf-8")
    assert "perplexity" in content
    assert "lm-evaluation-harness" in content
    assert "29.6 (measured)" in content


def test_compare_cli_includes_reporting_comparison(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    gates = tmp_path / "gates.yaml"
    out = tmp_path / "comparison.json"
    baseline.write_text(
        json.dumps(
            _report_payload(
                "baseline",
                likelihood={"status": "pass", "perplexity": 8.0},
                metric_references={},
            )
        ),
        encoding="utf-8",
    )
    current.write_text(
        json.dumps(
            _report_payload(
                "current",
                likelihood={"status": "pass", "perplexity": 10.0},
                metric_references={},
            )
        ),
        encoding="utf-8",
    )
    gates.write_text("likelihood:\n  perplexity_max: 9.0\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "compare",
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--gates",
            str(gates),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["gates"]["overall_status"] == "warning"
    assert payload["reporting_comparison"]["biggest_regressions"][0]["path"] == (
        "likelihood.perplexity"
    )


def test_reference_comparison_cli_writes_transposed_table(tmp_path):
    report = tmp_path / "qwen.json"
    report.write_text(
        json.dumps(
            _report_payload(
                "qwen",
                likelihood={"status": "pass", "perplexity": 29.6},
                metric_references={},
            )
        ),
        encoding="utf-8",
    )
    out = tmp_path / "comparison_transposed.md"

    result = CliRunner().invoke(
        app,
        [
            "report",
            "reference-comparison",
            "--report",
            f"Qwen={report}",
            "--out",
            str(out),
            "--format",
            "markdown-transposed",
        ],
    )

    assert result.exit_code == 0, result.output
    content = out.read_text(encoding="utf-8")
    assert "# Metric Values" in content
    assert "| Model | likelihood.nll | likelihood.perplexity |" in content
    assert "# Metric References" in content


def test_reference_comparison_cli_can_include_all_coverage_metrics(tmp_path):
    report = tmp_path / "qwen.json"
    report.write_text(
        json.dumps(
            _report_payload(
                "qwen",
                likelihood={"status": "pass", "perplexity": 29.6, "num_tokens": 12},
                metric_coverage={
                    "entries": [
                        {"section": "likelihood", "metric": "perplexity", "status": "measured"},
                        {"section": "likelihood", "metric": "num_tokens", "status": "measured"},
                    ]
                },
                metric_references={},
            )
        ),
        encoding="utf-8",
    )
    out = tmp_path / "comparison_all.md"

    result = CliRunner().invoke(
        app,
        [
            "report",
            "reference-comparison",
            "--report",
            f"Qwen={report}",
            "--out",
            str(out),
            "--format",
            "markdown-transposed",
            "--all-metrics",
        ],
    )

    assert result.exit_code == 0, result.output
    content = out.read_text(encoding="utf-8")
    assert "likelihood.perplexity" in content
    assert "likelihood.num_tokens" in content
    assert "likelihood.token_logprobs" in content


def test_reference_comparison_cli_all_metrics_discovers_nested_report_metrics(tmp_path):
    report = tmp_path / "qwen.json"
    report.write_text(
        json.dumps(
            _report_payload(
                "qwen",
                likelihood={"status": "pass", "perplexity": 29.6, "num_tokens": 12},
                generation={
                    "status": "pass",
                    "json_validity": {"json_valid": True},
                    "ngram_repetition": {"repetition_rate": 0.0, "unique_ngram_ratio": 1.0},
                },
                serving={
                    "status": "pass",
                    "batch_invariance": {
                        "mean_kl": 0.0,
                        "p95_kl": 0.0,
                        "top10_overlap": 1.0,
                    },
                },
                metric_references={},
            )
        ),
        encoding="utf-8",
    )
    out = tmp_path / "comparison_all_nested.html"

    result = CliRunner().invoke(
        app,
        [
            "report",
            "reference-comparison",
            "--report",
            f"Qwen={report}",
            "--out",
            str(out),
            "--format",
            "html-by-model",
            "--all-metrics",
        ],
    )

    assert result.exit_code == 0, result.output
    content = out.read_text(encoding="utf-8")
    assert "token_logprobs" in content
    assert "blocked" in content
    assert "not_collected" in content
    assert "batch_invariance.mean_kl" in content
    assert "batch_mean_kl" in content
    assert "generation.json_validity.json_valid" in content
    assert "json_valid</h3>" in content
    assert "ngram_repetition.unique_ngram_ratio" in content
    assert "ngram_repetition_rate" in content
    assert "serving.batch_invariance.mean_kl" in content
    assert "Evidence / reason" in content
    assert "Required inputs" in content
    assert "metric-grid" in content
    assert "show not_collected catalog gaps" in content
    assert "Mean KL drift for the same prompt alone versus inside a batch." in content
    assert "Boolean result of parsing generated text as JSON." in content


def test_reference_comparison_cli_writes_csv(tmp_path):
    report = tmp_path / "qwen.json"
    report.write_text(
        json.dumps(
            _report_payload(
                "qwen",
                likelihood={"status": "pass", "perplexity": 29.6},
                metric_references={},
            )
        ),
        encoding="utf-8",
    )
    out = tmp_path / "comparison.csv"

    result = CliRunner().invoke(
        app,
        [
            "report",
            "reference-comparison",
            "--report",
            f"Qwen={report}",
            "--out",
            str(out),
            "--format",
            "csv",
        ],
    )

    assert result.exit_code == 0, result.output
    content = out.read_text(encoding="utf-8")
    assert content.startswith("section,metric,reference,degradation_rule,Qwen")
    assert "likelihood,perplexity,lm-evaluation-harness" in content


def test_reference_comparison_cli_writes_html_dashboard(tmp_path):
    report = tmp_path / "qwen.json"
    report.write_text(
        json.dumps(
            _report_payload(
                "qwen",
                likelihood={"status": "pass", "perplexity": 29.6, "num_tokens": 12},
                serving={"status": "warning", "batch_mean_kl": 0.1},
                metric_coverage={
                    "entries": [
                        {"section": "likelihood", "metric": "perplexity", "status": "measured"},
                        {"section": "likelihood", "metric": "num_tokens", "status": "measured"},
                        {"section": "serving", "metric": "batch_mean_kl", "status": "measured"},
                    ]
                },
                metric_references={},
            )
        ),
        encoding="utf-8",
    )
    out = tmp_path / "comparison.html"

    result = CliRunner().invoke(
        app,
        [
            "report",
            "reference-comparison",
            "--report",
            f"Qwen={report}",
            "--out",
            str(out),
            "--format",
            "html-dashboard",
            "--all-metrics",
        ],
    )

    assert result.exit_code == 0, result.output
    content = out.read_text(encoding="utf-8")
    assert "<!doctype html>" in content
    assert "OVIQS Target Model Quality Dashboard" in content
    assert 'class="model-button active"' in content
    assert 'data-model="qwen"' in content
    assert 'id="critical-comparison"' in content
    assert "likelihood.perplexity" in content
    assert "Report:" in content
    assert "Evidence / reason" in content
    assert "deterministic drift oracle" in content
    assert ">fail</span>" in content
    assert '<body class="hide-not-collected">' in content


def test_reference_comparison_cli_writes_html_by_model_for_multiple_reports(tmp_path):
    qwen = tmp_path / "qwen.json"
    mistral = tmp_path / "mistral.json"
    qwen.write_text(
        json.dumps(
            _report_payload(
                "qwen",
                likelihood={"status": "pass", "perplexity": 29.6},
                serving={"status": "warning", "batch_mean_kl": 0.1},
                metric_references={},
            )
        ),
        encoding="utf-8",
    )
    mistral.write_text(
        json.dumps(
            _report_payload(
                "mistral",
                likelihood={"status": "pass", "perplexity": 7.8},
                serving={"status": "pass", "batch_mean_kl": 0.0},
                metric_references={},
            )
        ),
        encoding="utf-8",
    )
    out = tmp_path / "comparison_by_model.html"

    result = CliRunner().invoke(
        app,
        [
            "report",
            "reference-comparison",
            "--report",
            f"Qwen={qwen}",
            "--report",
            f"Mistral={mistral}",
            "--out",
            str(out),
            "--format",
            "html-by-model",
        ],
    )

    assert result.exit_code == 0, result.output
    content = out.read_text(encoding="utf-8")
    assert 'data-model="qwen"' in content
    assert 'data-model="mistral"' in content
    assert 'id="model-panel-qwen"' in content
    assert 'id="model-panel-mistral"' in content
    assert "Critical Metrics Comparison" in content
