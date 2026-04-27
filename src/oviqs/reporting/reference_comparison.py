from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oviqs.references import get_metric_reference

DEFAULT_METRICS = (
    ("likelihood", "nll"),
    ("likelihood", "perplexity"),
    ("likelihood", "bits_per_byte"),
    ("inference_equivalence", "mean_kl"),
    ("inference_equivalence", "p95_kl"),
    ("inference_equivalence", "top1_changed_rate"),
    ("generation", "json_validity"),
    ("generation", "schema_validity"),
    ("generation", "repetition_rate"),
    ("long_context", "lost_in_middle_score"),
    ("long_context", "distractor_sensitivity"),
    ("long_context", "authoritative_margin"),
    ("long_context", "contradiction_rate"),
    ("serving", "batch_mean_kl"),
    ("serving", "batch_p95_kl"),
    ("serving", "batch_top1_changed_rate"),
    ("serving", "kv_mean_kl"),
    ("serving", "kv_top1_change_rate"),
    ("rag", "context_precision"),
    ("rag", "context_recall"),
    ("rag", "faithfulness"),
    ("agent", "tool_call_validity"),
    ("agent", "recovery_score"),
    ("performance", "forward_latency_ms_mean"),
    ("performance", "generation_latency_ms"),
)

SECTION_DESCRIPTIONS = {
    "likelihood": (
        "Language-model likelihood on standard text. Lower NLL/PPL means the model assigns "
        "higher probability to the reference corpus."
    ),
    "inference_equivalence": (
        "Runtime numerical equivalence. These metrics compare logits distributions across "
        "devices or execution variants."
    ),
    "long_context": (
        "Controlled context-use diagnostics: context gain, position sensitivity, distractors "
        "and conflicting evidence."
    ),
    "generation": (
        "Application-level output checks: structure, repetition, entity preservation and "
        "contract compliance."
    ),
    "serving": (
        "Serving correctness under batching and cached decode. The same prompt should keep "
        "stable logits and generated prefixes."
    ),
    "rag": (
        "Retrieval and grounding checks with labeled evidence, citations and answer support."
    ),
    "agent": (
        "Trace-level checks for tool calls, state, recovery, grounding and policy constraints."
    ),
    "performance": "Same-hardware latency and throughput measurements.",
}

SECTION_ORDER = (
    "likelihood",
    "inference_equivalence",
    "long_context",
    "generation",
    "serving",
    "rag",
    "agent",
    "performance",
)

METRIC_EXPLANATIONS = {
    "nll": "Mean negative log-likelihood. Lower is better.",
    "perplexity": "exp(NLL). Lower means better corpus prediction.",
    "bits_per_byte": "Byte-normalized language-model score. Lower is better.",
    "mean_kl": "Mean KL divergence between reference and current token distributions.",
    "p95_kl": "95th percentile KL; highlights tail runtime drift.",
    "top1_changed_rate": "Fraction of positions where the top-1 token changed.",
    "json_validity": "1 means the output parses as JSON; 0 means invalid JSON.",
    "schema_validity": "1 means required structure/schema checks passed.",
    "repetition_rate": "Repeated n-gram fraction. Lower is better.",
    "lost_in_middle_score": "Positive means middle context is worse than edges.",
    "distractor_sensitivity": "NLL change caused by distractor context.",
    "authoritative_margin": "Authoritative answer log-prob minus best conflicting answer log-prob.",
    "contradiction_rate": "Fraction of contradiction/conflict failures.",
    "batch_mean_kl": "Mean drift between single-sample and batched inference.",
    "batch_p95_kl": "Tail drift between single-sample and batched inference.",
    "batch_top1_changed_rate": "Top-1 instability introduced by batching.",
    "kv_mean_kl": "Mean drift between full-forward logits and cached decode logits.",
    "kv_top1_change_rate": "Top-1 instability in cached decode.",
    "context_precision": "Fraction of retrieved contexts that are relevant.",
    "context_recall": "Fraction of expected evidence covered by retrieved contexts.",
    "faithfulness": "Fraction of answer claims supported by retrieved evidence.",
    "tool_call_validity": "Fraction of tool calls satisfying tool schema/policy checks.",
    "recovery_score": "Fraction of tool-error scenarios recovered successfully.",
    "forward_latency_ms_mean": "Mean full-forward latency on the measured hardware.",
    "generation_latency_ms": "End-to-end generation latency on the measured hardware.",
}


@dataclass(frozen=True)
class ReportInput:
    label: str
    path: Path


def parse_report_inputs(values: list[str]) -> list[ReportInput]:
    reports = []
    for value in values:
        label, sep, path = value.partition("=")
        if sep:
            reports.append(ReportInput(label=label, path=Path(path)))
        else:
            path_obj = Path(value)
            reports.append(ReportInput(label=path_obj.stem, path=path_obj))
    return reports


def build_reference_comparison(
    report_inputs: list[ReportInput],
    metrics: list[tuple[str, str]] | None = None,
    include_all_metrics: bool = False,
) -> dict[str, Any]:
    reports = [(item.label, _load_json(item.path), str(item.path)) for item in report_inputs]
    metric_specs = (
        collect_metric_coverage_specs([report for _label, report, _path in reports])
        if include_all_metrics
        else metrics or list(DEFAULT_METRICS)
    )
    rows = []
    for section, metric in metric_specs:
        row: dict[str, Any] = {
            "section": section,
            "metric": metric,
            **_reference_columns(reports, section, metric),
        }
        for label, report, _path in reports:
            value_path, value = find_metric_value(report.get(section, {}), metric)
            row[label] = format_metric_value(value)
            row[f"{label}_status"] = metric_status(report, section, metric, value)
            row[f"{label}_path"] = f"{section}.{value_path}" if value_path else ""
        rows.append(row)
    return {
        "reports": [{"label": label, "path": path} for label, _report, path in reports],
        "rows": rows,
    }


def collect_metric_coverage_specs(reports: list[dict[str, Any]]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    seen = set()
    for report in reports:
        entries = report.get("metric_coverage", {}).get("entries", [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            section = entry.get("section")
            metric = entry.get("metric")
            if not isinstance(section, str) or not isinstance(metric, str):
                continue
            key = (section, metric)
            if key in seen:
                continue
            seen.add(key)
            specs.append(key)
    if specs:
        return specs
    return list(DEFAULT_METRICS)


def render_reference_comparison_markdown(comparison: dict[str, Any]) -> str:
    labels = [item["label"] for item in comparison["reports"]]
    headers = [
        "Section",
        "Metric",
        "Reference",
        "Degradation rule",
        *labels,
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in comparison["rows"]:
        values = [
            row["section"],
            row["metric"],
            row.get("reference", ""),
            row.get("degradation_rule", ""),
            *[status_value(row, label) for label in labels],
        ]
        lines.append("| " + " | ".join(_escape_md(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def render_transposed_reference_comparison_markdown(comparison: dict[str, Any]) -> str:
    labels = [item["label"] for item in comparison["reports"]]
    metric_headers = [f"{row['section']}.{row['metric']}" for row in comparison["rows"]]
    lines = [
        "# Metric Values",
        "",
        "| Model | " + " | ".join(metric_headers) + " |",
        "|" + "|".join(["---"] * (len(metric_headers) + 1)) + "|",
    ]
    for label in labels:
        values = [status_value(row, label) for row in comparison["rows"]]
        lines.append("| " + " | ".join(_escape_md(value) for value in [label, *values]) + " |")

    lines.extend(
        [
            "",
            "# Metric References",
            "",
            "| Metric | Reference | Degradation rule |",
            "|---|---|---|",
        ]
    )
    for row in comparison["rows"]:
        metric = f"{row['section']}.{row['metric']}"
        lines.append(
            "| "
            + " | ".join(
                _escape_md(value)
                for value in [
                    metric,
                    row.get("reference", ""),
                    row.get("degradation_rule", ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_reference_comparison(
    comparison: dict[str, Any],
    out: Path,
    output_format: str,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        payload = json.dumps(comparison, indent=2, ensure_ascii=False) + "\n"
        out.write_text(payload, encoding="utf-8")
        return
    if output_format == "markdown":
        out.write_text(render_reference_comparison_markdown(comparison), encoding="utf-8")
        return
    if output_format == "markdown-transposed":
        out.write_text(
            render_transposed_reference_comparison_markdown(comparison),
            encoding="utf-8",
        )
        return
    if output_format == "html-dashboard":
        out.write_text(render_reference_comparison_html_dashboard(comparison), encoding="utf-8")
        return
    if output_format == "csv":
        labels = [item["label"] for item in comparison["reports"]]
        fieldnames = ["section", "metric", "reference", "degradation_rule", *labels]
        with out.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in comparison["rows"]:
                writer.writerow(
                    {
                        "section": row["section"],
                        "metric": row["metric"],
                        "reference": row.get("reference", ""),
                        "degradation_rule": row.get("degradation_rule", ""),
                        **{label: status_value(row, label) for label in labels},
                    }
                )
        return
    raise ValueError(f"Unsupported reference comparison format: {output_format}")


def render_reference_comparison_html_dashboard(comparison: dict[str, Any]) -> str:
    labels = [item["label"] for item in comparison["reports"]]
    rows_by_section = _rows_by_section(comparison["rows"])
    nav_sections = [section for section in SECTION_ORDER if section in rows_by_section]
    nav_sections.extend(section for section in rows_by_section if section not in nav_sections)
    cards = "\n".join(_overview_card(label, comparison["rows"]) for label in labels)
    sections = "\n".join(
        _section_html(section, rows_by_section[section], labels) for section in nav_sections
    )
    nav = "\n".join(
        f'<a href="#{_html_id(section)}">{html.escape(section.replace("_", " ").title())}</a>'
        for section in nav_sections
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OVIQS Target Model Quality Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --panel-soft: #f1f4f8;
      --text: #16202a;
      --muted: #5d6b7a;
      --line: #d8dee8;
      --accent: #0068b5;
      --bad: #b42318;
      --warn: #b54708;
      --good: #067647;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --sans: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      line-height: 1.45;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(255,255,255,.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(8px);
    }}
    .topbar {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 16px 24px 12px;
    }}
    h1 {{ margin: 0 0 10px; font-size: 22px; letter-spacing: 0; }}
    nav {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    nav a {{
      color: var(--accent);
      text-decoration: none;
      border: 1px solid var(--line);
      background: var(--panel-soft);
      border-radius: 6px;
      padding: 5px 9px;
      font-size: 13px;
    }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 20px 24px 48px; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .card, section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
    }}
    .card {{ padding: 14px; }}
    .card h2 {{ margin: 0 0 10px; font-size: 16px; }}
    .card dl {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 6px 12px;
      margin: 0;
      font-size: 13px;
    }}
    .card dt {{ color: var(--muted); }}
    .card dd {{ margin: 0; font-family: var(--mono); }}
    section {{ margin-top: 16px; overflow: hidden; }}
    .section-head {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-soft);
    }}
    .section-head h2 {{ margin: 0 0 4px; font-size: 18px; }}
    .section-head p {{ margin: 0; color: var(--muted); font-size: 13px; }}
    .metric {{ border-top: 1px solid var(--line); }}
    .metric:first-of-type {{ border-top: 0; }}
    details.metric summary {{
      cursor: pointer;
      list-style: none;
      padding: 12px 16px;
      display: grid;
      grid-template-columns: minmax(210px, 300px) 1fr;
      gap: 16px;
      align-items: start;
    }}
    details.metric summary::-webkit-details-marker {{ display: none; }}
    .metric-title strong {{ display: block; font-size: 14px; }}
    .metric-title span {{ color: var(--muted); font-size: 12px; }}
    .values {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 8px;
    }}
    .value {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      padding: 8px;
      min-width: 0;
    }}
    .model {{ color: var(--muted); font-size: 12px; margin-bottom: 2px; }}
    .number {{
      font-family: var(--mono);
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .status {{ font-size: 11px; margin-top: 3px; color: var(--muted); }}
    .status.measured {{ color: var(--good); }}
    .status.section_warning {{ color: var(--warn); }}
    .status.section_fail, .status.unknown {{ color: var(--bad); }}
    .details-body {{
      border-top: 1px dashed var(--line);
      padding: 12px 16px 16px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      color: var(--muted);
      font-size: 13px;
    }}
    .details-body h3 {{ margin: 0 0 6px; font-size: 13px; color: var(--text); }}
    .details-body p {{ margin: 0 0 8px; }}
    code {{
      font-family: var(--mono);
      background: var(--panel-soft);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 1px 4px;
    }}
    @media (max-width: 800px) {{
      details.metric summary, .details-body {{ grid-template-columns: 1fr; }}
      main, .topbar {{ padding-left: 14px; padding-right: 14px; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <h1>OVIQS Target Model Quality Dashboard</h1>
      <nav>{nav}</nav>
    </div>
  </header>
  <main>
    <div class="summary-grid">{cards}</div>
    {sections}
  </main>
</body>
</html>
"""


def _rows_by_section(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["section"]), []).append(row)
    return grouped


def _overview_card(label: str, rows: list[dict[str, Any]]) -> str:
    measured = 0
    warning = 0
    unknown = 0
    key_rows: list[dict[str, Any]] = []
    key_metrics = {
        ("likelihood", "perplexity"),
        ("generation", "json_validity"),
        ("serving", "batch_mean_kl"),
        ("serving", "kv_mean_kl"),
        ("performance", "generation_latency_ms"),
    }
    for row in rows:
        status = str(row.get(f"{label}_status", ""))
        if status == "measured":
            measured += 1
        elif status.startswith("unknown"):
            unknown += 1
        else:
            warning += 1
        if (row["section"], row["metric"]) in key_metrics:
            key_rows.append(row)

    key_items = "\n".join(
        f"<dt>{_h(row['metric'])}</dt><dd>{_h(status_value(row, label) or 'n/a')}</dd>"
        for row in key_rows
    )
    return f"""
      <article class="card">
        <h2>{_h(label)}</h2>
        <dl>
          <dt>Measured</dt><dd>{measured}</dd>
          <dt>Warnings</dt><dd>{warning}</dd>
          <dt>Unknown</dt><dd>{unknown}</dd>
          {key_items}
        </dl>
      </article>
    """


def _section_html(section: str, rows: list[dict[str, Any]], labels: list[str]) -> str:
    title = section.replace("_", " ").title()
    description = SECTION_DESCRIPTIONS.get(section, "Metric group from the input reports.")
    metrics = "\n".join(_metric_html(row, labels) for row in rows)
    return f"""
      <section id="{_html_id(section)}">
        <div class="section-head">
          <h2>{_h(title)}</h2>
          <p>{_h(description)}</p>
        </div>
        {metrics}
      </section>
    """


def _metric_html(row: dict[str, Any], labels: list[str]) -> str:
    metric = str(row["metric"])
    metric_id = f"{row['section']}.{metric}"
    explanation = _metric_explanation(row)
    values = "\n".join(_metric_value_html(row, label) for label in labels)
    paths = "\n".join(
        f"<dt>{_h(label)}</dt><dd><code>{_h(row.get(f'{label}_path') or 'not found')}</code></dd>"
        for label in labels
    )
    sources = _sources_html(row)
    oracle = row.get("oracle") or "Use the listed reference and degradation rule."
    return f"""
        <details class="metric">
          <summary>
            <span class="metric-title">
              <strong>{_h(metric_id)}</strong>
              <span>{_h(explanation)}</span>
            </span>
            <span class="values">{values}</span>
          </summary>
          <div class="details-body">
            <div>
              <h3>Reference</h3>
              <p><strong>Primary:</strong> {_h(row.get('reference') or 'n/a')}</p>
              <p><strong>Oracle:</strong> {_h(oracle)}</p>
              <p><strong>Degradation:</strong> {_h(row.get('degradation_rule') or 'n/a')}</p>
              {sources}
            </div>
            <div>
              <h3>Report Paths</h3>
              <dl>{paths}</dl>
            </div>
          </div>
        </details>
    """


def _metric_value_html(row: dict[str, Any], label: str) -> str:
    value = row.get(label) or "n/a"
    status = str(row.get(f"{label}_status", "unknown"))
    return f"""
              <span class="value">
                <span class="model">{_h(label)}</span>
                <span class="number">{_h(value)}</span>
                <span class="status {_status_class(status)}">{_h(status)}</span>
              </span>
    """


def _sources_html(row: dict[str, Any]) -> str:
    sources = row.get("sources")
    if not isinstance(sources, list) or not sources:
        return ""
    items = []
    for source in sources[:4]:
        if not isinstance(source, dict):
            continue
        name = _h(source.get("name") or source.get("url") or "source")
        url = source.get("url")
        if isinstance(url, str) and url:
            items.append(f'<li><a href="{_h(url)}">{name}</a></li>')
        else:
            items.append(f"<li>{name}</li>")
    if not items:
        return ""
    return "<h3>Sources</h3><ul>" + "".join(items) + "</ul>"


def _metric_explanation(row: dict[str, Any]) -> str:
    metric = str(row["metric"])
    if metric in METRIC_EXPLANATIONS:
        return METRIC_EXPLANATIONS[metric]
    oracle = row.get("oracle")
    if isinstance(oracle, str) and oracle:
        return oracle
    reference = row.get("reference")
    if isinstance(reference, str) and reference:
        return f"Measured against {reference}."
    return metric.replace("_", " ").capitalize()


def _status_class(status: str) -> str:
    if status == "measured":
        return "measured"
    if status.startswith("unknown"):
        return "unknown"
    if status.startswith("section_warning"):
        return "section_warning"
    return "section_fail"


def _html_id(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-")


def _h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _reference_columns(
    reports: list[tuple[str, dict[str, Any], str]],
    section: str,
    metric: str,
) -> dict[str, str]:
    for _label, report, _path in reports:
        reference = find_metric_reference(report, section, metric)
        if reference:
            return {
                "reference": str(reference.get("primary_reference", "")),
                "degradation_rule": str(reference.get("degradation_rule", "")),
                "oracle": str(reference.get("oracle", "")),
                "sources": reference.get("sources", []),
            }
    reference_obj = get_metric_reference(metric)
    if reference_obj is None:
        return {"reference": "", "degradation_rule": ""}
    return {
        "reference": reference_obj.primary_reference,
        "degradation_rule": reference_obj.degradation_rule,
        "oracle": reference_obj.oracle,
        "sources": [source.to_dict() for source in reference_obj.sources],
    }


def find_metric_reference(
    report: dict[str, Any],
    section: str,
    metric: str,
) -> dict[str, Any] | None:
    refs = report.get("metric_references", {})
    if not isinstance(refs, dict):
        return None
    section_refs = refs.get(section, {})
    if not isinstance(section_refs, dict):
        return None
    for key, value in section_refs.items():
        if not isinstance(value, dict):
            continue
        metric_names = value.get("metric_names", [])
        if key == metric or key.endswith(f".{metric}") or metric in metric_names:
            return value
    return None


def find_metric_value(payload: Any, metric: str, prefix: str = "") -> tuple[str | None, Any]:
    if not isinstance(payload, dict):
        return None, None
    if metric in payload:
        return metric, payload[metric]
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            found_path, found_value = find_metric_value(value, metric, path)
            if found_path:
                return found_path, found_value
    return None, None


def metric_status(
    report: dict[str, Any],
    section: str,
    metric: str,
    value: Any,
) -> str:
    if value is None:
        reason = _coverage_reason(report, section, metric)
        return f"unknown ({reason})" if reason else "unknown"
    section_payload = report.get(section)
    section_status = section_payload.get("status") if isinstance(section_payload, dict) else None
    if section_status and section_status not in {"pass", "measured"}:
        return f"section_{section_status}"
    return "measured"


def format_metric_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def status_value(row: dict[str, Any], label: str) -> str:
    value = row.get(label, "")
    status = row.get(f"{label}_status", "")
    if not value:
        return status
    return f"{value} ({status})"


def _coverage_reason(report: dict[str, Any], section: str, metric: str) -> str:
    coverage = report.get("metric_coverage", {}).get("entries", [])
    if not isinstance(coverage, list):
        return ""
    for item in coverage:
        if item.get("section") == section and item.get("metric") == metric:
            return str(item.get("reason") or "")
    return ""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
