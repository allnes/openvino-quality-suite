from __future__ import annotations

from collections import defaultdict

import numpy as np


def sample_length_bucket(num_tokens: int) -> str:
    if num_tokens <= 4096:
        return "0_4k"
    if num_tokens <= 16384:
        return "4_16k"
    if num_tokens <= 32768:
        return "16_32k"
    if num_tokens <= 65536:
        return "32_64k"
    if num_tokens <= 131072:
        return "64_128k"
    return "128k_plus"


def effective_context_bucket(left_context_tokens: int) -> str:
    if left_context_tokens <= 1024:
        return "ctx_0_1k"
    if left_context_tokens <= 4096:
        return "ctx_1_4k"
    if left_context_tokens <= 16384:
        return "ctx_4_16k"
    if left_context_tokens <= 32768:
        return "ctx_16_32k"
    if left_context_tokens <= 65536:
        return "ctx_32_64k"
    return "ctx_64k_plus"


def relative_position_bucket(pos: int, seq_len: int) -> str:
    r = pos / max(seq_len - 1, 1)
    if r < 0.10:
        return "0_10"
    if r < 0.30:
        return "10_30"
    if r < 0.50:
        return "30_50"
    if r < 0.70:
        return "50_70"
    if r < 0.90:
        return "70_90"
    return "90_100"


def aggregate_position_bucketed_ppl(per_token: list[dict], seq_len: int) -> dict[str, dict]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for item in per_token:
        bucket = relative_position_bucket(int(item["absolute_pos"]), seq_len)
        buckets[bucket].append(float(item["nll"]))
    return {
        bucket: {
            "nll": float(np.mean(values)),
            "ppl": float(np.exp(np.mean(values))),
            "tokens": len(values),
        }
        for bucket, values in buckets.items()
    }
