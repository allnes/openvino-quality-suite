# ruff: noqa: E501

from __future__ import annotations

import json
from html import escape
from typing import Any

from oviqs.domain.reporting import ReportViewModel
from oviqs.domain.reporting.paths import section_title


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
      --bg: #f7f8fa;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #5d6673;
      --line: #d8dee6;
      --pass: #147a3f;
      --warn: #946200;
      --fail: #b42318;
      --unknown: #5d6673;
    }}
    body {{ margin: 0; font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; background: var(--bg); color: var(--text); }}
    header {{ padding: 24px 32px 16px; background: var(--panel); border-bottom: 1px solid var(--line); }}
    main {{ padding: 20px 32px 40px; }}
    h1 {{ margin: 0 0 12px; font-size: 24px; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; }}
    .meta, .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
    .tile, details {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }}
    .label {{ color: var(--muted); font-size: 12px; }}
    .value {{ font-weight: 650; overflow-wrap: anywhere; }}
    .status-pass {{ color: var(--pass); }}
    .status-warning {{ color: var(--warn); }}
    .status-fail {{ color: var(--fail); }}
    .status-unknown {{ color: var(--unknown); }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); }}
    th, td {{ padding: 7px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ font-size: 12px; color: var(--muted); background: #eef2f6; cursor: pointer; }}
    input {{ width: min(520px, 100%); box-sizing: border-box; padding: 9px 10px; border: 1px solid var(--line); border-radius: 6px; }}
    summary {{ cursor: pointer; font-weight: 650; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
  </style>
</head>
<body>
<header>
  <h1>OVIQS Report: {escape(str(run.get("id", "unknown")))}</h1>
  <div class="meta">
    {_tile("Overall", str(summary.get("overall_status", "unknown")).upper(), summary.get("overall_status", "unknown"))}
    {_tile("Model", run.get("model"))}
    {_tile("Reference", run.get("reference"))}
    {_tile("Current", run.get("current"))}
    {_tile("Device", run.get("device"))}
    {_tile("Precision", run.get("precision"))}
    {_tile("Suite", run.get("suite"))}
    {_tile("Created", run.get("created_at"))}
    {_tile("Gates", f"{summary.get('passed', 0)} pass / {summary.get('warning', 0)} warn / {summary.get('failed', 0)} fail / {summary.get('unknown', 0)} unknown")}
  </div>
</header>
<main>
  <h2>Top Findings</h2>
  <table>
    <thead><tr><th>Severity</th><th>Status</th><th>Finding</th><th>Recommendation</th></tr></thead>
    <tbody>{"".join(_finding_row(item) for item in findings) or '<tr><td colspan="4">No findings.</td></tr>'}</tbody>
  </table>
  <h2>Biggest Regressions</h2>
  <table>
    <thead><tr><th>Metric</th><th>Current</th><th>Baseline</th><th>Delta</th><th>Status</th></tr></thead>
    <tbody>{"".join(_regression_row(item) for item in regressions) or '<tr><td colspan="5">No baseline regressions.</td></tr>'}</tbody>
  </table>
  <h2>Diagnostic Cards</h2>
  <div class="cards">{"".join(_section_card(section, payload) for section, payload in sections.items())}</div>
  <h2>Status Chart</h2>
  <canvas id="statusChart" width="720" height="180" aria-label="Metric status counts chart"></canvas>
  <h2>Diagnostic Sections</h2>
  {"".join(_section_details(section, metrics) for section in sections)}
  <h2>Sample-Level Outliers</h2>
  <table>
    <thead><tr><th>Section</th><th>Sample</th><th>Metric</th><th>Value</th><th>Mean</th><th>Distance</th></tr></thead>
    <tbody>{"".join(_outlier_row(item) for item in outliers) or '<tr><td colspan="6">No numeric sample outliers detected.</td></tr>'}</tbody>
  </table>
  <h2>Metric Table</h2>
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
    return (
        f'<div class="tile"><div class="label">{escape(section_title(section))}</div>'
        f'<div class="value status-{status}">{status.upper()}</div>'
        f"<div>{payload['total']} metrics, {payload['unknown']} unknown</div>"
        f"<div><strong>Key:</strong> {escape(_key_metrics(metrics))}</div>"
        f"<div><strong>Thresholds:</strong> {escape(_threshold_summary(metrics))}</div>"
        f"<div><strong>Delta:</strong> {escape(_delta_summary(metrics))}</div>"
        f"<div>{escape(interpretation)}</div>"
        f'<div><a href="#section-{escape(section)}">details</a></div></div>'
    )


def _section_details(section: str, metrics: list[dict[str, Any]]) -> str:
    rows = [metric for metric in metrics if metric.get("section") == section]
    body = "".join(_metric_row(item) for item in rows)
    return (
        f'<details id="section-{escape(section)}">'
        f"<summary>{escape(section_title(section))}</summary>"
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
