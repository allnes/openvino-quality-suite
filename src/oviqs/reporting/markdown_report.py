from __future__ import annotations

from typing import Any


def render_markdown_report(report: dict[str, Any]) -> str:
    run = report.get("run", {})
    summary = report.get("summary", {})
    lines = [
        f"# OVIQS Report: {run.get('id', 'unknown')}",
        "",
        "## Overall status",
        "",
        str(summary.get("overall_status", "unknown")).upper(),
        "",
    ]
    findings = summary.get("main_findings") or []
    if findings:
        lines.extend(["## Main findings", ""])
        lines.extend(f"{idx}. {finding}" for idx, finding in enumerate(findings, start=1))
        lines.append("")
    for section in [
        "inference_equivalence",
        "likelihood",
        "long_context",
        "rag",
        "agent",
        "serving",
        "performance",
    ]:
        payload = report.get(section) or {}
        if not payload:
            continue
        lines.extend(
            [f"## {section.replace('_', ' ').title()}", "", "| Metric | Value |", "|---|---:|"]
        )
        for key, value in payload.items():
            if isinstance(value, int | float | str):
                lines.append(f"| {key} | {value} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
