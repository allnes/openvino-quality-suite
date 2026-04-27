from __future__ import annotations

import numpy as np


def context_gain(nll_by_context: dict[str, float], baseline_key: str = "0k") -> dict[str, float]:
    base = nll_by_context[baseline_key]
    return {k: float(base - v) for k, v in nll_by_context.items() if k != baseline_key}


def context_saturation_curve(
    nll_by_context: dict[int, float], baseline_length: int = 0
) -> dict[str, dict]:
    base = nll_by_context[baseline_length]
    return {
        str(length): {
            "nll": float(nll),
            "ppl": float(np.exp(nll)),
            "context_gain": float(base - nll),
        }
        for length, nll in sorted(nll_by_context.items())
        if length != baseline_length
    }


def lost_in_middle_score_from_ppl(position_bucketed_ppl: dict[str, float]) -> float:
    edges = np.mean([position_bucketed_ppl["0_10"], position_bucketed_ppl["90_100"]])
    middle = np.mean([position_bucketed_ppl["30_50"], position_bucketed_ppl["50_70"]])
    return float(middle / max(edges, 1e-12) - 1.0)


def lost_in_middle_score_from_quality(position_score: dict[str, float]) -> float:
    edges = np.mean([position_score["0_10"], position_score["90_100"]])
    middle = np.mean([position_score["30_50"], position_score["50_70"]])
    return float(1.0 - middle / max(edges, 1e-12))


def degradation_slope(length_to_quality: dict[int, float]) -> float:
    if len(length_to_quality) < 2:
        raise ValueError("degradation_slope requires at least two lengths")
    xs = [np.log2(length) for length in length_to_quality]
    ys = list(length_to_quality.values())
    slope, _intercept = np.polyfit(xs, ys, deg=1)
    return float(slope)


def distractor_sensitivity(nll_clean: float, nll_distracted: float) -> float:
    return float(nll_distracted - nll_clean)


def conflict_sensitivity(nll_clean: float, nll_conflicted: float) -> float:
    """Return the NLL increase caused by conflicting context.

    Positive values mean the conflict made the target less likely. Negative values are
    possible when the model overfits to the conflicting setup or the supplied target is
    easier under the conflicted prompt.
    """

    return float(nll_conflicted - nll_clean)


def authoritative_margin(candidate_logprobs: dict[str, float], authoritative_key: str) -> float:
    authoritative = candidate_logprobs[authoritative_key]
    conflicting = [v for k, v in candidate_logprobs.items() if k != authoritative_key]
    if not conflicting:
        raise ValueError("Need at least one conflicting candidate")
    return float(authoritative - max(conflicting))


def conflict_entropy(candidate_logprobs: dict[str, float]) -> float:
    """Entropy over candidate answer probabilities in a conflict test.

    Higher entropy means the model is less decisive among candidate answers. Inputs are
    log-probabilities for mutually exclusive candidate answers.
    """

    if not candidate_logprobs:
        raise ValueError("conflict_entropy requires at least one candidate")
    values = np.asarray(list(candidate_logprobs.values()), dtype=np.float64)
    values = values - np.max(values)
    probs = np.exp(values)
    probs = probs / np.sum(probs)
    return float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))
