from __future__ import annotations

from pathlib import Path

from oviqs.reporting.reference_comparison import (
    build_reference_comparison,
    parse_report_inputs,
    write_reference_comparison,
)


class ReferenceComparisonAdapter:
    def write(
        self,
        reports: list[str],
        out: Path,
        format_name: str,
        include_all_metrics: bool,
    ) -> None:
        comparison = build_reference_comparison(
            parse_report_inputs(reports),
            include_all_metrics=include_all_metrics,
        )
        write_reference_comparison(comparison, out, format_name)
