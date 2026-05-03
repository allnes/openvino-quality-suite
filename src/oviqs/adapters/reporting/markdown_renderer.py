from __future__ import annotations

from typing import Any

from oviqs.domain.reporting import ReportViewModel
from oviqs.domain.reporting.paths import section_title


class MarkdownReportRenderer:
    format_name = "markdown"

    def render(self, report: ReportViewModel) -> str:
        payload = report.report
        analysis = report.analysis.to_dict()
        metrics = analysis.get("metrics", [])
        lines = _executive_summary(payload, analysis)
        lines.extend(_report_findings(payload))
        lines.extend(_analysis_findings(analysis))
        lines.extend(_gate_results(payload))
        lines.extend(_diagnostic_cards(metrics))
        lines.extend(_biggest_regressions(analysis))
        lines.extend(_reproducibility(payload))
        lines.extend(_sample_outliers(analysis))
        lines.extend(_metric_details(metrics))
        lines.extend(_debug_appendix(payload, metrics))
        return "\n".join(lines).rstrip() + "\n"


def _executive_summary(payload: dict[str, Any], analysis: dict[str, Any]) -> list[str]:
    run = payload.get("run", {})
    summary = analysis.get("summary", {})
    return [
        f"# OVIQS Report: {run.get('id', 'unknown')}",
        "",
        "## Executive Summary",
        "",
        f"- Overall status: **{str(summary.get('overall_status', 'unknown')).upper()}**",
        f"- Model: `{run.get('model') or 'n/a'}`",
        f"- Reference: `{run.get('reference') or 'n/a'}`",
        f"- Current: `{run.get('current') or 'n/a'}`",
        f"- Device: `{run.get('device') or 'n/a'}`",
        f"- Precision: `{run.get('precision') or 'n/a'}`",
        f"- Suite: `{run.get('suite') or 'n/a'}`",
        f"- Created: `{run.get('created_at') or 'n/a'}`",
        (
            "- Gates: "
            f"{summary.get('passed', 0)} pass / "
            f"{summary.get('warning', 0)} warning / "
            f"{summary.get('failed', 0)} fail / "
            f"{summary.get('unknown', 0)} unknown"
        ),
        "",
    ]


def _report_findings(payload: dict[str, Any]) -> list[str]:
    main_findings = payload.get("summary", {}).get("main_findings") or []
    if not main_findings:
        return []
    lines = ["## Report Findings", ""]
    lines.extend(f"{idx}. {finding}" for idx, finding in enumerate(main_findings, start=1))
    lines.append("")
    return lines


def _analysis_findings(analysis: dict[str, Any]) -> list[str]:
    findings = analysis.get("findings", [])[:5]
    if not findings:
        return []
    lines = ["## Top Analysis Findings", ""]
    lines.extend(["| Severity | Status | Finding | Recommendation |", "|---|---|---|---|"])
    for finding in findings:
        lines.append(
            "| "
            f"{finding.get('severity', '')} | "
            f"{finding.get('status', '')} | "
            f"{_escape_table(str(finding.get('title', '')))} | "
            f"{_escape_table(str(finding.get('recommendation', '')))} |"
        )
    lines.append("")
    return lines


def _gate_results(payload: dict[str, Any]) -> list[str]:
    gate_rows = _gate_rows(payload.get("gates"))
    if not gate_rows:
        return []
    lines = ["## Gate Results", ""]
    lines.extend(
        [
            "| Section | Gate | Metric Path | Value | Threshold | Status | Reference |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for row in gate_rows:
        lines.append(
            f"| {section_title(row['section'])} | `{row['gate']}` | "
            f"`{row['metric_path']}` | {_fmt(row['value'])} | "
            f"{_fmt(row['threshold'])} | {row['status']} | {row['reference_status']} |"
        )
    lines.append("")
    return lines


def _diagnostic_cards(metrics: list[dict[str, Any]]) -> list[str]:
    lines = ["## Diagnostic Cards", ""]
    lines.extend(
        [
            (
                "| Section | Status | Key Metrics | Thresholds | Baseline Delta | "
                "Interpretation | Details |"
            ),
            "|---|---|---|---|---|---|---|",
        ]
    )
    for section, rows in _group_metrics(metrics).items():
        details_anchor = section.replace("_", "-")
        lines.append(
            f"| {section_title(section)} | {_section_status(rows)} | "
            f"{_escape_table(_key_metrics(rows))} | "
            f"{_escape_table(_threshold_summary(rows))} | "
            f"{_escape_table(_delta_summary(rows))} | "
            f"{_escape_table(_section_interpretation(section, rows))} | "
            f"[details](#{details_anchor}) |"
        )
    lines.append("")
    return lines


def _biggest_regressions(analysis: dict[str, Any]) -> list[str]:
    regressions = analysis.get("biggest_regressions", [])
    if not regressions:
        return []
    lines = ["## Biggest Regressions", ""]
    lines.extend(["| Metric | Current | Baseline | Delta | Status |", "|---|---:|---:|---:|---|"])
    for metric in regressions[:10]:
        lines.append(
            f"| `{metric.get('path')}` | {_fmt(metric.get('value'))} | "
            f"{_fmt(metric.get('baseline_value'))} | {_fmt(metric.get('delta_abs'))} | "
            f"{metric.get('status')} |"
        )
    lines.append("")
    return lines


def _reproducibility(payload: dict[str, Any]) -> list[str]:
    reproducibility = payload.get("reproducibility")
    if not isinstance(reproducibility, dict) or not reproducibility:
        return []
    lines = ["## Reproducibility", ""]
    lines.extend(["| Field | Value |", "|---|---|"])
    for key, value in _flat_rows(reproducibility):
        lines.append(f"| `{_escape_table(key)}` | {_escape_table(_fmt(value))} |")
    lines.append("")
    return lines


def _sample_outliers(analysis: dict[str, Any]) -> list[str]:
    outliers = analysis.get("sample_outliers", [])
    if not outliers:
        return []
    lines = ["## Sample-Level Outliers", ""]
    lines.extend(
        ["| Section | Sample | Metric | Value | Mean | Distance |", "|---|---|---|---:|---:|---:|"]
    )
    for item in outliers[:20]:
        sample_id = item.get("sample_id") or item.get("sample_index")
        lines.append(
            f"| {item.get('section')} | {sample_id} | "
            f"{item.get('metric')} | {_fmt(item.get('value'))} | "
            f"{_fmt(item.get('mean'))} | {_fmt(item.get('distance'))} |"
        )
    lines.append("")
    return lines


def _metric_details(metrics: list[dict[str, Any]]) -> list[str]:
    lines = ["## Metric Details", ""]
    lines.extend(
        [
            "| Path | Value | Baseline | Delta | Threshold | Status | Reference | Rule |",
            "|---|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for metric in metrics:
        lines.append(
            f"| `{metric.get('path')}` | {_fmt(metric.get('value'))} | "
            f"{_fmt(metric.get('baseline_value'))} | {_fmt(metric.get('delta_abs'))} | "
            f"{_fmt(metric.get('threshold'))} | {metric.get('status')} | "
            f"{_escape_table(str(metric.get('reference_id') or 'n/a'))} | "
            f"{_escape_table(str(_metric_rule(metric)))} |"
        )
    lines.append("")
    return lines


def _debug_appendix(payload: dict[str, Any], metrics: list[dict[str, Any]]) -> list[str]:
    lines = ["## Developer Debug Appendix", ""]
    for section, rows in _group_metrics(metrics).items():
        section_payload = payload.get(section) or {}
        if not rows and not section_payload:
            continue
        lines.extend([f"### {section_title(section)}", ""])
        lines.append(f"- Raw JSON path: `{section}`")
        if isinstance(section_payload, dict) and section_payload.get("warnings"):
            lines.append(f"- Warnings: `{section_payload.get('warnings')}`")
        evidence = [row.get("path") for row in rows[:8]]
        if evidence:
            lines.append("- Evidence paths: " + ", ".join(f"`{item}`" for item in evidence))
        lines.append("")
    return lines


def _group_metrics(metrics: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        grouped.setdefault(str(metric.get("section") or "unknown"), []).append(metric)
    return grouped


def _gate_rows(gates: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(gates, dict):
        return rows
    sections = gates.get("sections")
    if not isinstance(sections, dict):
        return rows
    for section, payload in sections.items():
        checks = payload.get("checks") if isinstance(payload, dict) else None
        if not isinstance(checks, dict):
            continue
        for gate_name, check in checks.items():
            if not isinstance(check, dict):
                continue
            rows.append(
                {
                    "section": str(section),
                    "gate": str(gate_name),
                    "metric_path": check.get("metric_path") or check.get("metric") or "n/a",
                    "value": check.get("value"),
                    "threshold": check.get("threshold"),
                    "status": check.get("status", "unknown"),
                    "reference_status": check.get("reference_status", "unknown"),
                }
            )
    return rows


def _flat_rows(payload: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(_flat_rows(value, path))
        elif isinstance(value, list):
            rows.append((path, ", ".join(_fmt(item) for item in value)))
        else:
            rows.append((path, value))
    return rows


def _section_status(rows: list[dict[str, Any]]) -> str:
    statuses = {row.get("status") for row in rows}
    if "fail" in statuses:
        return "fail"
    if "warning" in statuses:
        return "warning"
    if "unknown" in statuses:
        return "unknown"
    return "pass"


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


def _section_interpretation(section: str, rows: list[dict[str, Any]]) -> str:
    unknown = sum(1 for row in rows if row.get("status") == "unknown")
    warning = sum(1 for row in rows if row.get("status") == "warning")
    fail = sum(1 for row in rows if row.get("status") == "fail")
    if fail:
        return f"{fail} failing metric(s); inspect evidence before accepting the run."
    if warning:
        return f"{warning} warning metric(s); compare against baseline and gates."
    if unknown:
        return f"{unknown} unknown metric(s); missing evidence is not a pass."
    key_values = ", ".join(f"{row.get('name')}={_fmt(row.get('value'))}" for row in rows[:3])
    return key_values or f"{section} metrics are present."


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "n/a"
    return str(value)


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _metric_rule(metric: dict[str, Any]) -> str:
    return str(metric.get("degradation_rule") or metric.get("threshold_rule") or "n/a")


__all__ = ["MarkdownReportRenderer"]
