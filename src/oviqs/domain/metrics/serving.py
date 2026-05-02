from __future__ import annotations

from oviqs.domain.metrics.distribution_drift import (
    aggregate_drift,
    distribution_drift,
    top1_changed_rate,
)


def batch_invariance_drift(alone_logits, batched_logits) -> dict[str, float]:
    """Compare logits for a sample alone and inside a batch."""

    drift = distribution_drift(alone_logits, batched_logits)
    out = aggregate_drift(drift)
    out["top1_changed_rate"] = top1_changed_rate(alone_logits, batched_logits)
    return out


def generation_prefix_divergence(alone_text: str, batched_text: str) -> dict[str, float | int]:
    """Measure deterministic generation prefix divergence for serving checks."""

    alone_tokens = alone_text.split()
    batched_tokens = batched_text.split()
    limit = min(len(alone_tokens), len(batched_tokens))
    matching = 0
    for idx in range(limit):
        if alone_tokens[idx] != batched_tokens[idx]:
            break
        matching += 1
    denom = max(max(len(alone_tokens), len(batched_tokens)), 1)
    return {
        "prefix_match_tokens": matching,
        "alone_tokens": len(alone_tokens),
        "batched_tokens": len(batched_tokens),
        "prefix_divergence_rate": 1.0 - matching / denom,
    }


def kv_cache_drift(full_logits, cached_logits) -> dict[str, float]:
    """Compare full-forward logits with cached decode logits when both are available."""

    drift = distribution_drift(full_logits, cached_logits)
    out = aggregate_drift(drift)
    out["top1_changed_rate"] = top1_changed_rate(full_logits, cached_logits)
    return out


def kv_cache_drift_interface() -> dict[str, None | list[str]]:
    return {
        "kv_cache_drift": None,
        "warnings": ["KV-cache drift requires a stateful runner implementation"],
    }
