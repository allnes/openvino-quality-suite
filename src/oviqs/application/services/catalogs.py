from __future__ import annotations

from pathlib import Path

import yaml

from oviqs.domain.models import (
    ExportCommand,
    GenAIModelEntry,
    GenAIModelMatrix,
    export_plan,
)
from oviqs.domain.references import MetricReference, list_metric_references, references_for_family

DEFAULT_MATRIX_PATH = Path("configs/examples/genai_metric_models.yaml")


def load_genai_model_matrix(path: str | Path | None = None) -> GenAIModelMatrix:
    matrix_path = Path(path) if path is not None else DEFAULT_MATRIX_PATH
    if not matrix_path.exists():
        raise FileNotFoundError(f"GenAI model matrix not found: {matrix_path}")
    payload = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    return GenAIModelMatrix.model_validate(payload)


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


__all__ = [
    "DEFAULT_MATRIX_PATH",
    "genai_export_commands",
    "list_genai_models",
    "list_metric_reference_catalog",
    "load_genai_model_matrix",
]
