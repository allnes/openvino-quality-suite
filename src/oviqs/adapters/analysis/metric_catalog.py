from __future__ import annotations

from oviqs.domain.references import MetricReference, get_metric_reference


class MetricReferenceCatalog:
    def get_reference(self, metric_name: str) -> MetricReference | None:
        return get_metric_reference(metric_name)


__all__ = ["MetricReferenceCatalog"]
