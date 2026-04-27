from __future__ import annotations


def relative_drop(baseline: float, current: float) -> float:
    return float((baseline - current) / max(abs(baseline), 1e-12))
