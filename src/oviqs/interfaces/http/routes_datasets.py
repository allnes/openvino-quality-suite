from __future__ import annotations

from typing import Any


def build_datasets_router(fastapi: Any, container: Any) -> Any:
    router = fastapi.APIRouter(prefix="/v1/datasets", tags=["datasets"])

    @router.get("")
    def list_datasets() -> dict[str, Any]:
        return {"datasets": [], "reader": type(container.dataset_reader).__name__}

    return router


__all__ = ["build_datasets_router"]
