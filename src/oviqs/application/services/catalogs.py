from __future__ import annotations

from pathlib import Path

from oviqs.domain.models import (
    ExportCommand,
    GenAIModelEntry,
    GenAIModelMatrix,
    export_plan,
    load_genai_model_matrix,
)
from oviqs.domain.references import MetricReference, list_metric_references, references_for_family


def list_genai_models(
    config: Path,
    tier: str | None = None,
    metric: str | None = None,
    family: str | None = None,
) -> tuple[GenAIModelMatrix, list[tuple[str, GenAIModelEntry]]]:
    matrix = load_genai_model_matrix(config)
    return matrix, matrix.list_models(tier=tier, metric=metric, family=family)


def genai_export_commands(
    config: Path,
    model: str,
    output_root: Path,
    variants: list[str] | None,
) -> list[ExportCommand]:
    matrix = load_genai_model_matrix(config)
    return export_plan(matrix, model_id=model, output_root=output_root, variants=variants)


def list_metric_reference_catalog(family: str | None = None) -> list[MetricReference]:
    return list(references_for_family(family) if family else list_metric_references())
