from __future__ import annotations

from typing import Any


def build_models_router(fastapi: Any, container: Any) -> Any:
    router = fastapi.APIRouter(prefix="/v1/models", tags=["models"])

    @router.get("")
    def list_models() -> dict[str, Any]:
        return {"models": [], "registry": container.settings.get("model_registry", "local")}

    return router


__all__ = ["build_models_router"]
