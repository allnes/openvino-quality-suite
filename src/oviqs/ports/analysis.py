from __future__ import annotations

from typing import Any, Protocol

from oviqs.domain.references import MetricReference
from oviqs.domain.reporting import AnalysisFinding, MetricObservation


class MetricCatalogPort(Protocol):
    def get_reference(self, metric_name: str) -> MetricReference | None: ...


class AnalysisRulePort(Protocol):
    def analyze(
        self,
        report: dict[str, Any],
        metrics: list[MetricObservation],
    ) -> list[AnalysisFinding]: ...


class TrendStorePort(Protocol):
    def load_baseline(self, report: dict[str, Any]) -> dict[str, Any] | None: ...
    def load_history(self, report: dict[str, Any]) -> list[dict[str, Any]]: ...
    def append(self, report: dict[str, Any]) -> None: ...


class GateEvaluatorPort(Protocol):
    def evaluate(
        self,
        report: dict[str, Any],
        gate_payload: dict[str, Any] | None,
    ) -> dict[str, Any]: ...


__all__ = [
    "AnalysisRulePort",
    "GateEvaluatorPort",
    "MetricCatalogPort",
    "TrendStorePort",
]
