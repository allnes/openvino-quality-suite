from __future__ import annotations

from oviqs.domain.references.catalog import (
    MetricReference,
    ReferenceSource,
    build_report_reference_manifest,
    get_metric_reference,
    list_metric_references,
    references_for_family,
    require_metric_reference,
)
from oviqs.domain.references.catalog import (
    get_metric_reference as get_metric_reference_from_catalog,
)
from oviqs.domain.references.catalog import (
    list_metric_references as list_catalog_metric_references,
)
from oviqs.domain.references.catalog import (
    references_for_family as catalog_references_for_family,
)
from oviqs.domain.references.catalog import (
    require_metric_reference as require_catalog_metric_reference,
)
from oviqs.domain.references.oracles import (
    build_report_reference_manifest as build_oracle_reference_manifest,
)

__all__ = [
    "MetricReference",
    "ReferenceSource",
    "build_report_reference_manifest",
    "get_metric_reference",
    "list_metric_references",
    "references_for_family",
    "require_metric_reference",
    "build_oracle_reference_manifest",
    "catalog_references_for_family",
    "get_metric_reference_from_catalog",
    "list_catalog_metric_references",
    "require_catalog_metric_reference",
]
