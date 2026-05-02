from __future__ import annotations

from importlib import import_module
from typing import Any

from oviqs.interfaces.http.routes_datasets import build_datasets_router
from oviqs.interfaces.http.routes_models import build_models_router
from oviqs.interfaces.http.routes_runs import build_runs_router
from oviqs.platform.bootstrap import build_default_container


def create_app() -> Any:
    try:
        fastapi = import_module("fastapi")
    except ImportError as exc:
        raise RuntimeError(
            "HTTP interface requires the optional FastAPI dependency. "
            "Install FastAPI in the runtime environment before calling create_app()."
        ) from exc

    app = fastapi.FastAPI(title="OVIQS", version="0.1.0")
    container = build_default_container()
    app.include_router(build_runs_router(fastapi, container))
    app.include_router(build_models_router(fastapi, container))
    app.include_router(build_datasets_router(fastapi, container))

    return app
