from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oviqs.application.reporting.normalization import flatten_report_metrics
from oviqs.domain.reporting import MetricObservation
from oviqs.ports.analysis import TrendStorePort


@dataclass(frozen=True)
class ReportComparison:
    metrics: tuple[MetricObservation, ...]
    biggest_regressions: tuple[MetricObservation, ...]
    biggest_improvements: tuple[MetricObservation, ...]
    unknown_metrics: tuple[MetricObservation, ...]
    missing_metrics: tuple[str, ...]
    trend_points: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": [metric.to_dict() for metric in self.metrics],
            "biggest_regressions": [metric.to_dict() for metric in self.biggest_regressions],
            "biggest_improvements": [metric.to_dict() for metric in self.biggest_improvements],
            "unknown_metrics": [metric.to_dict() for metric in self.unknown_metrics],
            "missing_metrics": list(self.missing_metrics),
            "trend_points": list(self.trend_points),
        }


class ReportComparisonService:
    def __init__(self, trend_store: TrendStorePort | None = None) -> None:
        self.trend_store = trend_store

    def compare(
        self,
        current: dict[str, Any],
        baseline: dict[str, Any] | None = None,
        gates: dict[str, Any] | None = None,
    ) -> ReportComparison:
        history = self.trend_store.load_history(current) if self.trend_store else []
        if baseline is None and self.trend_store is not None:
            baseline = self.trend_store.load_baseline(current)
        metrics = tuple(flatten_report_metrics(current, baseline=baseline, gates=gates))
        numeric = [metric for metric in metrics if metric.delta_abs is not None]
        regressions = tuple(
            sorted(
                (metric for metric in numeric if (metric.delta_abs or 0.0) > 0.0),
                key=_delta_sort_key,
                reverse=True,
            )[:10]
        )
        improvements = tuple(
            sorted(
                (metric for metric in numeric if (metric.delta_abs or 0.0) < 0.0),
                key=_delta_sort_key,
            )[:10]
        )
        unknown = tuple(metric for metric in metrics if metric.status == "unknown")
        missing = tuple(metric.path for metric in unknown if metric.value is None)
        return ReportComparison(
            metrics=metrics,
            biggest_regressions=regressions,
            biggest_improvements=improvements,
            unknown_metrics=unknown,
            missing_metrics=missing,
            trend_points=tuple(_trend_points(history, metrics)),
        )


def _delta_sort_key(metric: MetricObservation) -> float:
    if metric.delta_rel is not None:
        return metric.delta_rel
    return float(metric.delta_abs or 0.0)


def _trend_points(
    history: list[dict[str, Any]],
    current_metrics: tuple[MetricObservation, ...],
) -> list[dict[str, Any]]:
    tracked_paths = {
        metric.path
        for metric in current_metrics
        if isinstance(metric.value, int | float) and not isinstance(metric.value, bool)
    }
    points: list[dict[str, Any]] = []
    for report in history[-20:]:
        run = report.get("run", {}) if isinstance(report, dict) else {}
        run_id = run.get("id") if isinstance(run, dict) else None
        for metric in flatten_report_metrics(report):
            if metric.path not in tracked_paths:
                continue
            if not isinstance(metric.value, int | float) or isinstance(metric.value, bool):
                continue
            points.append(
                {
                    "run_id": run_id,
                    "path": metric.path,
                    "value": metric.value,
                    "status": metric.status,
                }
            )
    return points


__all__ = ["ReportComparison", "ReportComparisonService"]
