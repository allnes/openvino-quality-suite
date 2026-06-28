# ruff: noqa: E501

from __future__ import annotations

import json
from html import escape
from typing import Any

from oviqs.domain.reporting import ReportViewModel
from oviqs.domain.reporting.paths import section_title

# Plain-language guide for each metric family: what it measures, the direction of
# "better", and a short intuition for an algorithms reader. Keyed by report
# section name. Used to turn raw metric tables into a self-explaining report.
FAMILY_GUIDE: dict[str, dict[str, str]] = {
    "inference_equivalence": {
        "what": "Per-position agreement between the PyTorch reference and the exported OpenVINO model on the same Intel GPU. This is the headline export/quantization-fidelity signal.",
        "watch": "mean_kl / p95_kl / max_kl (lower = closer), mean_logit_cosine (→1 is ideal), top1_changed_rate (fraction of positions where the argmax token flipped), top5/top10_overlap.",
        "better": "lower drift, higher overlap/cosine",
    },
    "likelihood": {
        "what": "How much probability the model assigns to ground-truth tokens under teacher forcing (causal NLL/PPL) on a fixed corpus.",
        "watch": "nll and perplexity (exp of mean NLL); word/byte-perplexity and bits_per_byte normalize across tokenizers.",
        "better": "lower NLL / perplexity",
    },
    "long_context": {
        "what": "Whether added context helps, saturates or hurts: context gain over length, lost-in-the-middle, distractor and conflict sensitivity.",
        "watch": "context_gain (higher = context helps), lost_in_middle_score (lower = no middle dip), degradation_slope, distractor_sensitivity.",
        "better": "higher gain, lower degradation",
    },
    "generation": {
        "what": "Observable properties of generated text: repetition, JSON/schema validity, entity preservation and topic drift.",
        "watch": "json_validity / schema_validity (1 = valid), repetition_rate (lower), unique_ngram_ratio (higher), entity_preservation_rate.",
        "better": "valid structure, low repetition",
    },
    "serving": {
        "what": "Stability across execution modes for the same prompt: single-vs-batch (batch invariance) and full-forward-vs-KV-cache decode.",
        "watch": "batch_mean_kl / batch_top1_changed_rate, kv_mean_kl / kv_top1_change_rate (all lower = more stable serving).",
        "better": "lower drift across modes",
    },
    "rag": {
        "what": "Retrieval-augmented quality split into retrieval ranking, evidence grounding and citation correctness.",
        "watch": "recall_at_k / mrr / ndcg / context_recall (higher), faithfulness and citation_precision/recall (higher), distractor_ratio (lower).",
        "better": "higher retrieval & grounding",
    },
    "agent": {
        "what": "Workflow quality of a structured agent trace: tool-call validity, redundancy, grounding, task completion and recovery.",
        "watch": "tool_call_validity / task_completion / recovery_score (higher), redundant_tool_call_rate / policy_violation_rate (lower).",
        "better": "higher validity & completion",
    },
    "performance": {
        "what": "Runtime cost on the target device; comparable only against a same-hardware baseline.",
        "watch": "forward_latency_ms_mean / p95 (lower), tokens_per_second_forward (higher), generation_latency_ms.",
        "better": "lower latency, higher throughput",
    },
}


class HtmlDashboardRenderer:
    format_name = "html-dashboard"

    def render(self, report: ReportViewModel) -> str:
        payload = report.report
        analysis = report.analysis.to_dict()
        run = payload.get("run", {})
        summary = analysis.get("summary", {})
        metrics = analysis.get("metrics", [])
        findings = analysis.get("findings", [])[:5]
        regressions = analysis.get("biggest_regressions", [])[:10]
        outliers = analysis.get("sample_outliers", [])[:20]
        sections = _section_cards(metrics)
        fidelity = _fidelity_panel(payload, metrics)
        data_json = json.dumps(
            {
                "metrics": metrics,
                "findings": findings,
                "biggest_regressions": regressions,
                "sample_outliers": outliers,
            },
            ensure_ascii=False,
        )
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OVIQS Report {escape(str(run.get("id", "unknown")))}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3f5f8;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #5d6673;
      --line: #d8dee6;
      --accent: #1f4e96;
      --pass: #147a3f;
      --warn: #946200;
      --fail: #b42318;
      --unknown: #5d6673;
    }}
    body {{ margin: 0; font: 14px/1.5 system-ui, -apple-system, Segoe UI, sans-serif; background: var(--bg); color: var(--text); }}
    header {{ padding: 24px 32px 18px; background: linear-gradient(180deg,#ffffff, #f7f9fc); border-bottom: 1px solid var(--line); }}
    main {{ padding: 20px 32px 48px; max-width: 1180px; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    h2 {{ margin: 30px 0 12px; font-size: 18px; border-left: 3px solid var(--accent); padding-left: 9px; }}
    .lede {{ color: var(--muted); margin: 0 0 14px; max-width: 760px; }}
    .meta, .cards, .guide {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; }}
    .tile, details, .card, .guide-item, .hero {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .02em; }}
    .value {{ font-weight: 650; overflow-wrap: anywhere; }}
    .hero {{ border-left: 4px solid var(--accent); padding: 16px 18px; margin: 4px 0 8px; }}
    .hero .nums {{ display: flex; flex-wrap: wrap; gap: 22px; margin-top: 10px; }}
    .hero .num b {{ display: block; font-size: 22px; }}
    .hero .num span {{ color: var(--muted); font-size: 12px; }}
    .pill {{ display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; font-weight: 650; border: 1px solid currentColor; }}
    .status-pass {{ color: var(--pass); }}
    .status-warning {{ color: var(--warn); }}
    .status-fail {{ color: var(--fail); }}
    .status-unknown {{ color: var(--unknown); }}
    .guide-item h3 {{ margin: 0 0 4px; font-size: 14px; }}
    .guide-item p {{ margin: 4px 0; color: var(--muted); font-size: 12.5px; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 14px; color: var(--muted); font-size: 12.5px; margin: 6px 0 2px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 7px 9px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ font-size: 12px; color: var(--muted); background: #eef2f6; cursor: pointer; }}
    tr:hover td {{ background: #f7fafe; }}
    input {{ width: min(520px, 100%); box-sizing: border-box; padding: 9px 10px; border: 1px solid var(--line); border-radius: 6px; }}
    summary {{ cursor: pointer; font-weight: 650; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    .card .desc {{ color: var(--muted); font-size: 12px; margin: 6px 0; }}
    .kv {{ font-size: 12.5px; margin: 2px 0; }}
  </style>
</head>
<body>
<header>
  <h1>OVIQS Report: {escape(str(run.get("id", "unknown")))}</h1>
  <p class="lede">OpenVINO Inference Quality Suite — diagnostic scorecard for an exported LLM. Sections below are <strong>evaluation surfaces</strong>; each shows what it measures, the direction of "better", and the measured numbers. The headline signal is <strong>inference equivalence</strong>: how closely the OpenVINO export tracks the PyTorch reference.</p>
  <div class="meta">
    {_tile("Overall", str(summary.get("overall_status", "unknown")).upper(), summary.get("overall_status", "unknown"))}
    {_tile("Model", run.get("model"))}
    {_tile("Reference", run.get("reference") or run.get("reference_model"))}
    {_tile("Current", run.get("current"))}
    {_tile("Device", run.get("device"))}
    {_tile("Precision", run.get("precision"))}
    {_tile("Suite", run.get("suite"))}
    {_tile("Created", run.get("created_at"))}
    {_tile("Gates", f"{summary.get('passed', 0)} pass / {summary.get('warning', 0)} warn / {summary.get('failed', 0)} fail / {summary.get('unknown', 0)} unknown")}
  </div>
</header>
<main>
  <p class="legend">
    <span><span class="pill status-pass">PASS</span> within gate</span>
    <span><span class="pill status-warning">WARN</span> needs review / evidence-limited</span>
    <span><span class="pill status-fail">FAIL</span> release-blocking</span>
    <span><span class="pill status-unknown">UNKNOWN</span> required evidence missing</span>
  </p>

  <h2>Export Fidelity (PyTorch &rarr; OpenVINO)</h2>
  {fidelity}

  <h2>What Each Surface Measures</h2>
  <div class="guide">{"".join(_guide_item(section) for section in sections)}</div>

  <h2>Top Findings</h2>
  <p class="lede">Ranked issues the analyzer raised, with the recommended next action.</p>
  <table>
    <thead><tr><th>Severity</th><th>Status</th><th>Finding</th><th>Recommendation</th></tr></thead>
    <tbody>{"".join(_finding_row(item) for item in findings) or '<tr><td colspan="4">No findings — all gated metrics passed.</td></tr>'}</tbody>
  </table>

  <h2>Biggest Regressions</h2>
  <p class="lede">Largest metric moves versus the baseline report (empty if no baseline was supplied).</p>
  <table>
    <thead><tr><th>Metric</th><th>Current</th><th>Baseline</th><th>Delta</th><th>Status</th></tr></thead>
    <tbody>{"".join(_regression_row(item) for item in regressions) or '<tr><td colspan="5">No baseline regressions.</td></tr>'}</tbody>
  </table>

  <h2>Diagnostic Cards</h2>
  <p class="lede">One card per surface: status, key measured values, and a plain-language read.</p>
  <div class="cards">{"".join(_section_card(section, payload) for section, payload in sections.items())}</div>

  <h2>Status Chart</h2>
  <canvas id="statusChart" width="720" height="180" aria-label="Metric status counts chart"></canvas>

  <h2>Diagnostic Sections</h2>
  {"".join(_section_details(section, metrics) for section in sections)}

  <h2>Sample-Level Outliers</h2>
  <p class="lede">Individual samples whose metric value is far from the section mean — useful for spotting a single bad prompt.</p>
  <table>
    <thead><tr><th>Section</th><th>Sample</th><th>Metric</th><th>Value</th><th>Mean</th><th>Distance</th></tr></thead>
    <tbody>{"".join(_outlier_row(item) for item in outliers) or '<tr><td colspan="6">No numeric sample outliers detected.</td></tr>'}</tbody>
  </table>

  <h2>Metric Table</h2>
  <p class="lede">Every scalar metric with its gate status, reference oracle and degradation rule. Type to filter.</p>
  <input id="q" aria-label="Search metrics" placeholder="Search metric path, status or reference">
  <table id="metrics">
    <thead><tr><th>Path</th><th>Value</th><th>Baseline</th><th>Delta</th><th>Status</th><th>Reference</th><th>Rule</th></tr></thead>
    <tbody>{"".join(_metric_row(item) for item in metrics)}</tbody>
  </table>

  <h2>Raw Embedded Data</h2>
  <p>
    <a href="report.json">report.json</a> ·
    <a href="analysis.json">analysis.json</a> ·
    <a href="metrics.csv">metrics.csv</a> ·
    <a href="sample_metrics.jsonl">sample_metrics.jsonl</a>
  </p>
  <details><summary>analysis data</summary><pre>{escape(data_json)}</pre></details>
</main>
<script>
const DATA = {data_json};
const statusCounts = DATA.metrics.reduce((acc, metric) => {{
  const key = metric.status || 'unknown';
  acc[key] = (acc[key] || 0) + 1;
  return acc;
}}, {{pass: 0, warning: 0, fail: 0, unknown: 0}});
const canvas = document.getElementById('statusChart');
const ctx = canvas.getContext('2d');
const chartItems = Object.entries(statusCounts);
const maxValue = Math.max(...chartItems.map(([, value]) => value), 1);
chartItems.forEach(([label, value], idx) => {{
  const x = 40 + idx * 160;
  const height = Math.round((value / maxValue) * 120);
  ctx.fillStyle = {{pass: '#147a3f', warning: '#946200', fail: '#b42318', unknown: '#5d6673'}}[label];
  ctx.fillRect(x, 140 - height, 80, height);
  ctx.fillStyle = '#17202a';
  ctx.fillText(label + ' ' + value, x, 160);
}});
const input = document.getElementById('q');
input.addEventListener('input', () => {{
  const q = input.value.toLowerCase();
  document.querySelectorAll('#metrics tbody tr').forEach(row => {{
    row.style.display = row.innerText.toLowerCase().includes(q) ? '' : 'none';
  }});
}});
document.querySelectorAll('th').forEach((th, idx) => {{
  th.addEventListener('click', () => {{
    const table = th.closest('table');
    const body = table.querySelector('tbody');
    [...body.querySelectorAll('tr')].sort((a, b) =>
      a.children[idx].innerText.localeCompare(b.children[idx].innerText, undefined, {{numeric: true}})
    ).forEach(row => body.appendChild(row));
  }});
}});
</script>
</body>
</html>
"""


def _tile(label: str, value: Any, status: Any = None) -> str:
    cls = f" status-{status}" if status in {"pass", "warning", "fail", "unknown"} else ""
    return (
        f'<div class="tile"><div class="label">{escape(label)}</div>'
        f'<div class="value{cls}">{escape(str(value or "n/a"))}</div></div>'
    )


def _guide_item(section: str) -> str:
    guide = FAMILY_GUIDE.get(section)
    if not guide:
        return ""
    return (
        f'<div class="guide-item"><h3>{escape(section_title(section))}</h3>'
        f'<p>{escape(guide["what"])}</p>'
        f'<p><strong>Watch:</strong> {escape(guide["watch"])}</p>'
        f'<p><strong>Better:</strong> {escape(guide["better"])}</p></div>'
    )


def _fidelity_panel(payload: dict[str, Any], metrics: list[dict[str, Any]]) -> str:
    """Headline panel for the OpenVINO-vs-PyTorch inference-equivalence section."""
    section = payload.get("inference_equivalence")
    if not isinstance(section, dict):
        return '<div class="hero"><div class="label">Inference equivalence</div><div class="value status-unknown">Not measured in this report.</div></div>'
    status = str(section.get("status", "unknown"))
    keys = [
        ("mean_kl", "Mean KL", "lower = closer"),
        ("p95_kl", "P95 KL", "tail divergence"),
        ("mean_logit_cosine", "Logit cosine", "→1 ideal"),
        ("top1_changed_rate", "Top-1 changed", "argmax flips"),
        ("top5_overlap", "Top-5 overlap", "higher better"),
    ]
    nums = []
    for key, label, hint in keys:
        if key in section and isinstance(section[key], int | float) and not isinstance(section[key], bool):
            nums.append(
                f'<div class="num"><b>{escape(_fmt(section[key]))}</b><span>{escape(label)} · {escape(hint)}</span></div>'
            )
    ref = payload.get("run", {}).get("reference_model") or payload.get("run", {}).get("reference") or "PyTorch reference"
    nums_html = "".join(nums) or '<div class="num"><span>No scalar drift values present.</span></div>'
    read = (
        "Export tracks the reference closely."
        if status == "pass"
        else "Inspect tail positions and top-k changes before trusting the export."
    )
    return (
        f'<div class="hero"><div class="label">PyTorch reference: <code>{escape(str(ref))}</code></div>'
        f'<div class="value status-{escape(status)}">{escape(status.upper())} — {escape(read)}</div>'
        f'<div class="nums">{nums_html}</div></div>'
    )


def _finding_row(item: dict[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{escape(str(item.get('severity', '')))}</td>"
        f"<td>{escape(str(item.get('status', '')))}</td>"
        f"<td>{escape(str(item.get('title', '')))}</td>"
        f"<td>{escape(str(item.get('recommendation', '')))}</td>"
        "</tr>"
    )


def _regression_row(item: dict[str, Any]) -> str:
    status = str(item.get("status", "unknown"))
    return (
        "<tr>"
        f"<td><code>{escape(str(item.get('path', '')))}</code></td>"
        f"<td>{escape(_fmt(item.get('value')))}</td>"
        f"<td>{escape(_fmt(item.get('baseline_value')))}</td>"
        f"<td>{escape(_fmt(item.get('delta_abs')))}</td>"
        f'<td class="status-{escape(status)}">{escape(status)}</td>'
        "</tr>"
    )


def _metric_row(item: dict[str, Any]) -> str:
    status = str(item.get("status", "unknown"))
    return (
        "<tr>"
        f"<td><code>{escape(str(item.get('path', '')))}</code></td>"
        f"<td>{escape(_fmt(item.get('value')))}</td>"
        f"<td>{escape(_fmt(item.get('baseline_value')))}</td>"
        f"<td>{escape(_fmt(item.get('delta_abs')))}</td>"
        f'<td class="status-{escape(status)}">{escape(status)}</td>'
        f"<td>{escape(str(item.get('reference_id') or 'n/a'))}</td>"
        f"<td>{escape(str(item.get('degradation_rule') or item.get('threshold_rule') or 'n/a'))}</td>"
        "</tr>"
    )


def _outlier_row(item: dict[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{escape(str(item.get('section', '')))}</td>"
        f"<td>{escape(str(item.get('sample_id') or item.get('sample_index') or ''))}</td>"
        f"<td>{escape(str(item.get('metric', '')))}</td>"
        f"<td>{escape(_fmt(item.get('value')))}</td>"
        f"<td>{escape(_fmt(item.get('mean')))}</td>"
        f"<td>{escape(_fmt(item.get('distance')))}</td>"
        "</tr>"
    )


def _section_cards(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    sections: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        section = str(metric.get("section") or "unknown")
        payload = sections.setdefault(
            section,
            {"total": 0, "warning": 0, "fail": 0, "unknown": 0, "metrics": []},
        )
        payload["total"] += 1
        payload["metrics"].append(metric)
        status = metric.get("status")
        if status in {"warning", "fail", "unknown"}:
            payload[status] += 1
    return sections


def _section_card(section: str, payload: dict[str, Any]) -> str:
    status = (
        "fail"
        if payload["fail"]
        else ("warning" if payload["warning"] else ("unknown" if payload["unknown"] else "pass"))
    )
    metrics = payload.get("metrics", [])
    interpretation = _section_interpretation(payload)
    guide = FAMILY_GUIDE.get(section, {})
    desc = (
        f'<div class="desc">{escape(guide["what"])}</div>' if guide.get("what") else ""
    )
    return (
        f'<div class="card"><div class="label">{escape(section_title(section))}</div>'
        f'<div class="value status-{status}">{status.upper()}</div>'
        f"{desc}"
        f'<div class="kv">{payload["total"]} metrics · {payload["unknown"]} unknown</div>'
        f'<div class="kv"><strong>Key:</strong> {escape(_key_metrics(metrics))}</div>'
        f'<div class="kv"><strong>Thresholds:</strong> {escape(_threshold_summary(metrics))}</div>'
        f'<div class="kv"><strong>Delta:</strong> {escape(_delta_summary(metrics))}</div>'
        f'<div class="kv">{escape(interpretation)}</div>'
        f'<div class="kv"><a href="#section-{escape(section)}">details</a></div></div>'
    )


def _section_details(section: str, metrics: list[dict[str, Any]]) -> str:
    rows = [metric for metric in metrics if metric.get("section") == section]
    body = "".join(_metric_row(item) for item in rows)
    guide = FAMILY_GUIDE.get(section, {})
    intro = (
        f"<p class=\"desc\">{escape(guide['what'])}</p>" if guide.get("what") else ""
    )
    return (
        f'<details id="section-{escape(section)}">'
        f"<summary>{escape(section_title(section))}</summary>"
        f"{intro}"
        "<table><thead><tr><th>Path</th><th>Value</th><th>Baseline</th><th>Delta</th>"
        "<th>Status</th><th>Reference</th><th>Rule</th></tr></thead>"
        f"<tbody>{body}</tbody></table></details>"
    )


def _section_interpretation(payload: dict[str, Any]) -> str:
    if payload["fail"]:
        return "Failing metrics require release-blocking review."
    if payload["warning"]:
        return "Warning metrics need baseline and gate inspection."
    if payload["unknown"]:
        return "Unknown metrics mean required evidence is missing."
    return "No warning, fail or unknown metrics in this section."


def _key_metrics(rows: list[dict[str, Any]]) -> str:
    ranked = sorted(rows, key=lambda row: _status_sort_key(str(row.get("status"))))
    values = [f"{row.get('name')}={_fmt(row.get('value'))}" for row in ranked[:3]]
    return ", ".join(values) or "n/a"


def _threshold_summary(rows: list[dict[str, Any]]) -> str:
    gated = [row for row in rows if row.get("threshold") is not None]
    if not gated:
        return "no gated metrics"
    counts: dict[str, int] = {}
    for row in gated:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))


def _delta_summary(rows: list[dict[str, Any]]) -> str:
    with_deltas = [
        row
        for row in rows
        if isinstance(row.get("delta_abs"), int | float)
        and not isinstance(row.get("delta_abs"), bool)
    ]
    if not with_deltas:
        return "no baseline"
    row = max(with_deltas, key=lambda item: abs(float(item["delta_abs"])))
    return f"{row.get('name')} {_fmt(row.get('baseline_value'))}->{_fmt(row.get('value'))}"


def _status_sort_key(status: str) -> int:
    return {"fail": 0, "warning": 1, "unknown": 2, "pass": 3}.get(status, 4)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "n/a"
    return str(value)


__all__ = ["HtmlDashboardRenderer"]
