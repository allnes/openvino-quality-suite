from __future__ import annotations

from html import escape

from oviqs.reporting.markdown_report import render_markdown_report


def render_html_report(report: dict) -> str:
    return "<pre>" + escape(render_markdown_report(report)) + "</pre>\n"
