from __future__ import annotations

from typing import Any

from oviqs.reporting.markdown_report import render_markdown_report


class MarkdownReportRenderer:
    format_name = "markdown"

    def render(self, report: dict[str, Any]) -> str:
        return render_markdown_report(report)
