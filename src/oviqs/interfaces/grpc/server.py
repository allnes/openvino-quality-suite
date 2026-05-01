from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from typing import Any


def create_server(*_services: Any) -> Any:
    try:
        grpc = import_module("grpc")
    except ImportError as exc:
        raise RuntimeError(
            "gRPC interface requires the optional grpcio dependency. "
            "Install grpcio in the runtime environment before calling create_server()."
        ) from exc
    return grpc.server(ThreadPoolExecutor(max_workers=4))
