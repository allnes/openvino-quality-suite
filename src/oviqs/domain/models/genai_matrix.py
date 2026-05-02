from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class GenAIModelEntry(BaseModel):
    id: str
    family: str
    size: str | None = None
    context: str | None = None
    minimal_weight_format: str | None = None
    non_distilled: bool = True
    openvino_artifact: str | None = None
    note: str | None = None
    metrics: list[str] = Field(default_factory=list)


class GenAIModelTier(BaseModel):
    default: str
    models: list[GenAIModelEntry] = Field(default_factory=list)


class ExportVariant(BaseModel):
    task: str
    weight_format: str
    backend: str


class GenAIModelMatrix(BaseModel):
    model_matrix: dict[str, GenAIModelTier]
    rag_support: dict[str, list[str]] = Field(default_factory=dict)
    export_variants: dict[str, ExportVariant] = Field(default_factory=dict)
    device_policy: dict[str, Any] = Field(default_factory=dict)

    def list_models(
        self,
        tier: str | None = None,
        metric: str | None = None,
        family: str | None = None,
    ) -> list[tuple[str, GenAIModelEntry]]:
        tiers = [tier] if tier else list(self.model_matrix)
        rows: list[tuple[str, GenAIModelEntry]] = []
        for tier_name in tiers:
            if tier_name not in self.model_matrix:
                raise KeyError(f"Unknown model tier: {tier_name}")
            for model in self.model_matrix[tier_name].models:
                if metric and metric not in model.metrics:
                    continue
                if family and family != model.family:
                    continue
                rows.append((tier_name, model))
        return rows

    def default_model(self, tier: str) -> GenAIModelEntry:
        if tier not in self.model_matrix:
            raise KeyError(f"Unknown model tier: {tier}")
        default_id = self.model_matrix[tier].default
        for model in self.model_matrix[tier].models:
            if model.id == default_id:
                return model
        raise ValueError(f"Default model {default_id!r} is not present in tier {tier!r}")

    def find_model(self, model_id: str) -> GenAIModelEntry:
        for _tier, model in self.list_models():
            if model.id == model_id:
                return model
        raise KeyError(f"Unknown model id: {model_id}")


class ExportCommand(BaseModel):
    model_id: str
    variant: str
    task: str
    weight_format: str
    backend: str
    output_dir: str
    command: list[str]

    @property
    def shell_command(self) -> str:
        return " ".join(self.command)


def export_plan(
    matrix: GenAIModelMatrix,
    model_id: str,
    output_root: str | Path = "models",
    variants: list[str] | None = None,
) -> list[ExportCommand]:
    matrix.find_model(model_id)
    selected_variants = variants or list(matrix.export_variants)
    commands: list[ExportCommand] = []
    for variant_name in selected_variants:
        if variant_name not in matrix.export_variants:
            raise KeyError(f"Unknown export variant: {variant_name}")
        variant = matrix.export_variants[variant_name]
        output_dir = Path(output_root) / f"{sanitize_model_name(model_id)}-{variant_name}"
        command = [
            "optimum-cli",
            "export",
            "openvino",
            "--model",
            model_id,
            "--task",
            variant.task,
            "--weight-format",
            variant.weight_format,
            str(output_dir),
        ]
        commands.append(
            ExportCommand(
                model_id=model_id,
                variant=variant_name,
                task=variant.task,
                weight_format=variant.weight_format,
                backend=variant.backend,
                output_dir=str(output_dir),
                command=command,
            )
        )
    return commands


def sanitize_model_name(model_id: str) -> str:
    return model_id.lower().replace("/", "--").replace("_", "-").replace(".", "-").replace(" ", "-")
