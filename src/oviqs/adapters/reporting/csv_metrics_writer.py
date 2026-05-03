from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from oviqs.domain.reporting import MetricObservation

FIELDS = [
    "path",
    "section",
    "name",
    "value",
    "unit",
    "status",
    "severity",
    "reference_id",
    "degradation_rule",
    "baseline_value",
    "delta_abs",
    "delta_rel",
    "threshold",
    "threshold_rule",
    "sample_count",
    "tags",
]


class CsvMetricsWriter:
    def render(self, metrics: list[MetricObservation]) -> str:
        rows = [_metric_row(metric) for metric in metrics]
        rendered = _render_with_pandas(rows)
        if rendered is not None:
            return rendered
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue()

    def write(self, metrics: list[MetricObservation], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(metrics), encoding="utf-8")


def _metric_row(metric: MetricObservation) -> dict[str, Any]:
    row = metric.to_dict()
    row["tags"] = ";".join(metric.tags)
    return row


def _render_with_pandas(rows: list[dict[str, Any]]) -> str | None:
    try:
        import pandas as pd
    except ImportError:
        return None
    return pd.DataFrame(rows, columns=FIELDS).to_csv(index=False)


__all__ = ["CsvMetricsWriter"]
