from __future__ import annotations


def attach_status(metrics: dict, status: str) -> dict:
    return {**metrics, "status": status}
