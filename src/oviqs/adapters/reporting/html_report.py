from __future__ import annotations

from html import escape

from oviqs.adapters.reporting.markdown_report import MarkdownReportRenderer


def render_html_report(report: dict) -> str:
    return "<pre>" + escape(MarkdownReportRenderer().render(report)) + "</pre>\n"


__all__ = ["render_html_report"]
