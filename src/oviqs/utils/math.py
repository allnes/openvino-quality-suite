from __future__ import annotations


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return default if denominator == 0 else numerator / denominator
