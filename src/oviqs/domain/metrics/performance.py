from __future__ import annotations


def tokens_per_second(num_tokens: int, latency_ms: float) -> float:
    return num_tokens / max(latency_ms / 1000.0, 1e-12)


__all__ = ["tokens_per_second"]
