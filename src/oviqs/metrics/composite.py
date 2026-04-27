from __future__ import annotations


def weighted_composite(scores: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(weights.get(k, 0.0) for k in scores)
    if total_weight <= 0:
        raise ValueError("No positive weights for supplied scores")
    return sum(scores[k] * weights.get(k, 0.0) for k in scores) / total_weight
