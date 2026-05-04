from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from oviqs.cli import app

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "reference" / "cli"
COMMANDS = [
    ("eval-likelihood", ["eval-likelihood"]),
    ("eval-drift", ["eval-drift"]),
    ("eval-long-context", ["eval-long-context"]),
    ("eval-serving", ["eval-serving"]),
    ("eval-rag", ["eval-rag"]),
    ("eval-agent", ["eval-agent"]),
    ("run-suite", ["run-suite"]),
    ("run-gpu-suite", ["run-gpu-suite"]),
    ("compare", ["compare"]),
    ("report-build", ["report", "build"]),
    ("report-analyze", ["report", "analyze"]),
    ("report-render", ["report", "render"]),
    ("report-metrics-table", ["report", "metrics-table"]),
    ("report-validate", ["report", "validate"]),
    ("report-reference-comparison", ["report", "reference-comparison"]),
    ("metric-long-context", ["metric-long-context"]),
    ("list-genai-models", ["list-genai-models"]),
    ("list-metric-references", ["list-metric-references"]),
    ("genai-export-plan", ["genai-export-plan"]),
]

CURATED_SECTIONS = {
    "eval-likelihood": [
        "## Example",
        "",
        "```bash",
        "oviq eval-likelihood \\",
        "  --model models/tiny-ov \\",
        "  --dataset data/likelihood.jsonl \\",
        "  --out reports/likelihood.json \\",
        "  --backend openvino-runtime \\",
        "  --device CPU",
        "```",
        "",
        "Use this command when aligned logits are available. If the selected backend cannot "
        "return token-level likelihood evidence, the affected metrics must stay `unknown` "
        "instead of being approximated from generated text.",
    ],
    "eval-drift": [
        "## Example",
        "",
        "```bash",
        "oviq eval-drift \\",
        "  --reference models/baseline-ov \\",
        "  --current models/current-ov \\",
        "  --dataset data/likelihood.jsonl \\",
        "  --out reports/drift.json \\",
        "  --reference-backend openvino-runtime \\",
        "  --current-backend openvino-runtime \\",
        "  --device CPU",
        "```",
        "",
        "Use drift checks when reference and current runs can be aligned over the same "
        "samples, tokenizer positions and metric definitions. If alignment cannot be "
        "proven, report the affected comparison as `unknown`.",
    ],
    "eval-long-context": [
        "## Example",
        "",
        "```bash",
        "oviq eval-long-context \\",
        "  --dataset data/long-context.jsonl \\",
        "  --model models/tiny-ov \\",
        "  --backend openvino-runtime \\",
        "  --lengths 4096,8192,16384 \\",
        "  --out reports/long-context.json",
        "```",
        "",
        "Keep the length set explicit so repeated CI runs compare the same context "
        "windows. Position and distractor metrics should stay separate from generic "
        "generation quality scores.",
    ],
    "eval-serving": [
        "## Example",
        "",
        "```bash",
        "oviq eval-serving \\",
        "  --model ovms://localhost:9000/models/tiny \\",
        "  --dataset data/serving.jsonl \\",
        "  --backend dummy \\",
        "  --device CPU \\",
        "  --out reports/serving.json",
        "```",
        "",
        "Serving diagnostics are for endpoint behavior such as batch invariance, prefix "
        "stability and cache-sensitive checks. Keep endpoint metadata in the report "
        "reproducibility section when publishing results.",
    ],
    "eval-rag": [
        "## Example",
        "",
        "```bash",
        "oviq eval-rag \\",
        "  --dataset data/rag-questions.jsonl \\",
        "  --answers reports/rag-answers.jsonl \\",
        "  --scorer placeholder \\",
        "  --out reports/rag.json",
        "```",
        "",
        "Use the placeholder scorer for deterministic fixture checks. Use optional RAG "
        "scorers only when their dependencies and judge inputs are available, and keep "
        "judge-based gaps visible as `unknown`.",
    ],
    "eval-agent": [
        "## Example",
        "",
        "```bash",
        "oviq eval-agent \\",
        "  --traces data/agent-traces.jsonl \\",
        "  --expected data/agent-expected.jsonl \\",
        "  --out reports/agent.json",
        "```",
        "",
        "Agent checks depend on trace structure. Keep tool calls, observations, final "
        "answers and expected state stable in fixtures so failures can be debugged from "
        "the report bundle without replaying a live agent.",
    ],
    "run-suite": [
        "## Example",
        "",
        "```bash",
        "oviq run-suite --config suites/local.yaml --out reports/current.json",
        "```",
        "",
        "The suite command is for orchestrated runs. Keep per-command experiments in the "
        "`eval-*` commands until their inputs and expected evidence are stable enough for "
        "a checked-in suite profile.",
    ],
    "run-gpu-suite": [
        "## Example",
        "",
        "```bash",
        "oviq run-gpu-suite \\",
        "  --model models/tiny-ov \\",
        "  --genai-model models/tiny-genai-ov \\",
        "  --dataset data/likelihood.jsonl \\",
        "  --out reports/gpu.json \\",
        "  --device GPU",
        "```",
        "",
        "Use `--model` for Runtime-backed logits checks and `--genai-model` for generation "
        "checks. A valid GPU run may still produce `unknown` metrics when a backend cannot "
        "expose the required evidence.",
    ],
    "compare": [
        "## Example",
        "",
        "```bash",
        "oviq compare \\",
        "  --baseline reports/baseline.json \\",
        "  --current reports/current.json \\",
        "  --gates configs/gates/default_gates.yaml \\",
        "  --out reports/comparison.json",
        "```",
        "",
        "Use this command for a compact comparison JSON. Prefer `oviq report analyze` or "
        "`oviq report build` when the review needs normalized findings, CSV metrics, "
        "Markdown or dashboard artifacts.",
    ],
    "report-build": [
        "## Example",
        "",
        "```bash",
        "oviq report build \\",
        "  --report reports/current.json \\",
        "  --baseline reports/baseline.json \\",
        "  --gates configs/gates.yaml \\",
        "  --out reports/current-bundle \\",
        "  --format all",
        "```",
        "",
        "The bundle is the publishable reporting artifact. It should contain the source "
        "report, normalized metrics, analysis output, metadata, and requested renderings.",
    ],
    "report-analyze": [
        "## Example",
        "",
        "```bash",
        "oviq report analyze \\",
        "  --report reports/current.json \\",
        "  --baseline reports/baseline.json \\",
        "  --gates configs/gates.yaml \\",
        "  --out reports/current.analysis.json",
        "```",
        "",
        "Analysis output is machine-readable. Use it in CI when the rendered dashboard is "
        "too heavy and the job only needs findings, gate results, and metric status.",
    ],
    "report-render": [
        "## Example",
        "",
        "```bash",
        "oviq report render \\",
        "  --bundle reports/current-bundle \\",
        "  --out reports/current-bundle/index.md \\",
        "  --format markdown",
        "```",
        "",
        "Render from an existing bundle so reports, metrics, analysis, and metadata stay "
        "consistent across Markdown and dashboard outputs.",
    ],
    "report-metrics-table": [
        "## Example",
        "",
        "```bash",
        "oviq report metrics-table \\",
        "  --report reports/current.json \\",
        "  --out reports/current.metrics.csv",
        "```",
        "",
        "Use the CSV for compact review, trend ingestion and spreadsheet-style diffs. "
        "Regenerate it from `report.json`; do not edit generated metric rows by hand.",
    ],
    "report-validate": [
        "## Example",
        "",
        "```bash",
        "oviq report validate --report reports/current.json",
        "```",
        "",
        "Validation checks the public report contract. It does not prove that every metric "
        "is meaningful; unsupported evidence should still be represented as `unknown`.",
    ],
    "report-reference-comparison": [
        "## Example",
        "",
        "```bash",
        "oviq report reference-comparison \\",
        "  --report baseline=reports/baseline.json \\",
        "  --report current=reports/current.json \\",
        "  --out reports/reference-comparison.md \\",
        "  --format markdown",
        "```",
        "",
        "Use labels when comparing multiple reports. The command lives under `oviq report`; "
        "legacy top-level comparison commands are intentionally not part of the new CLI.",
    ],
    "metric-long-context": [
        "## Example",
        "",
        "```bash",
        "oviq metric-long-context \\",
        '  --position-ppl-json \'{"start":2.1,"middle":3.4,"end":2.3}\'',
        "```",
        "",
        "This helper is a deterministic metric utility for quick checks and documentation "
        "examples. Full long-context reports should still be produced with "
        "`oviq eval-long-context`.",
    ],
    "list-genai-models": [
        "## Notes",
        "",
        "This command reports the model catalogue known to the current checkout. Treat it as "
        "a planning aid, then validate the exact exported model directory with a real "
        "evaluation before publishing a quality result.",
    ],
    "list-metric-references": [
        "## Example",
        "",
        "```bash",
        "oviq list-metric-references --family likelihood --json",
        "```",
        "",
        "Reference metadata tells reviewers which metrics are gateable and which evidence "
        "is required. Do not add gates for metrics that do not have a documented "
        "reference or oracle.",
    ],
    "genai-export-plan": [
        "## Example",
        "",
        "```bash",
        "oviq genai-export-plan --model TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "```",
        "",
        "The export plan is documentation for the operator. It does not replace an actual "
        "OpenVINO model export or a subsequent `oviq run-gpu-suite` validation run.",
    ],
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expected_pages = {f"{slug}.md" for slug, _command in COMMANDS} | {"index.md"}
    for stale_page in OUT_DIR.glob("*.md"):
        if stale_page.name not in expected_pages:
            stale_page.unlink()
    runner = CliRunner()
    index_rows = []
    for slug, command in COMMANDS:
        result = runner.invoke(app, [*command, "--help"], color=False)
        if result.exit_code != 0:
            raise RuntimeError(f"{' '.join(command)} --help failed:\n{result.output}")
        command_name = " ".join(["oviq", *command])
        (OUT_DIR / f"{slug}.md").write_text(
            "\n".join(
                [
                    f"# `{command_name}`",
                    "",
                    "Generated from the Typer command help.",
                    "",
                    "```text",
                    clean_help_output(result.output),
                    "```",
                    "",
                    *CURATED_SECTIONS.get(slug, []),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        index_rows.append(f"- [`{command_name}`]({slug}.md)")
    (OUT_DIR / "index.md").write_text(
        "# CLI reference\n\nGenerated from `oviq --help` and command-level help output.\n\n"
        + "\n".join(index_rows)
        + "\n",
        encoding="utf-8",
    )


def strip_ansi(value: str) -> str:
    plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    return plain.replace("Usage: root", "Usage: oviq")


def clean_help_output(value: str) -> str:
    return "\n".join(line.rstrip() for line in strip_ansi(value).splitlines()).rstrip()


if __name__ == "__main__":
    main()
