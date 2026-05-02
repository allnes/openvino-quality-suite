from __future__ import annotations

import numpy as np
from scipy.special import log_softmax


def distribution_drift(ref_logits, cur_logits, eps: float = 1e-12) -> dict[str, np.ndarray]:
    """Compute KL, JS, entropy drift and logit cosine per position.

    KL is `KL(P_ref || P_cur)`. Inputs are logits with identical shape and are converted
    to float32 before log-softmax for numerical stability.
    """

    ref_arr = np.asarray(ref_logits, dtype=np.float32)
    cur_arr = np.asarray(cur_logits, dtype=np.float32)
    if ref_arr.shape != cur_arr.shape:
        raise ValueError(
            f"Cannot compute drift for shape mismatch: {ref_arr.shape} vs {cur_arr.shape}"
        )
    if ref_arr.ndim < 2:
        raise ValueError(f"Expected logits ending in vocab dimension, got {ref_arr.shape}")

    ref_logp = log_softmax(ref_arr, axis=-1)
    cur_logp = log_softmax(cur_arr, axis=-1)
    ref_p = np.exp(ref_logp)
    cur_p = np.exp(cur_logp)

    kl = np.sum(ref_p * (ref_logp - cur_logp), axis=-1)
    m = 0.5 * (ref_p + cur_p)
    log_m = np.log(np.maximum(m, eps))
    js = 0.5 * np.sum(ref_p * (ref_logp - log_m), axis=-1) + 0.5 * np.sum(
        cur_p * (cur_logp - log_m), axis=-1
    )

    ref_entropy = -np.sum(ref_p * ref_logp, axis=-1)
    cur_entropy = -np.sum(cur_p * cur_logp, axis=-1)

    dot = np.sum(ref_arr * cur_arr, axis=-1)
    ref_norm = np.linalg.norm(ref_arr, axis=-1)
    cur_norm = np.linalg.norm(cur_arr, axis=-1)
    cosine = dot / np.maximum(ref_norm * cur_norm, eps)

    return {
        "kl_per_pos": kl.astype(np.float32),
        "js_per_pos": js.astype(np.float32),
        "ref_entropy_per_pos": ref_entropy.astype(np.float32),
        "cur_entropy_per_pos": cur_entropy.astype(np.float32),
        "entropy_drift_per_pos": (cur_entropy - ref_entropy).astype(np.float32),
        "logit_cosine_per_pos": cosine.astype(np.float32),
    }


def aggregate_drift(drift: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "mean_kl": float(np.mean(drift["kl_per_pos"])),
        "p95_kl": float(np.percentile(drift["kl_per_pos"], 95)),
        "max_kl": float(np.max(drift["kl_per_pos"])),
        "mean_js": float(np.mean(drift["js_per_pos"])),
        "p95_js": float(np.percentile(drift["js_per_pos"], 95)),
        "mean_entropy_drift": float(np.mean(drift["entropy_drift_per_pos"])),
        "mean_logit_cosine": float(np.mean(drift["logit_cosine_per_pos"])),
    }


def topk_overlap(ref_logits, cur_logits, k: int = 10) -> float:
    ref_arr = np.asarray(ref_logits)
    cur_arr = np.asarray(cur_logits)
    if ref_arr.shape != cur_arr.shape:
        raise ValueError(
            f"Cannot compute top-k overlap for mismatch: {ref_arr.shape} vs {cur_arr.shape}"
        )
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, ref_arr.shape[-1])
    ref_top = np.argpartition(ref_arr, -k, axis=-1)[..., -k:]
    cur_top = np.argpartition(cur_arr, -k, axis=-1)[..., -k:]
    overlaps = []
    for ref_set, cur_set in zip(ref_top.reshape(-1, k), cur_top.reshape(-1, k), strict=True):
        overlaps.append(len(set(ref_set.tolist()) & set(cur_set.tolist())) / k)
    return float(np.mean(overlaps))


def top1_changed_rate(ref_logits, cur_logits) -> float:
    ref_top = np.argmax(np.asarray(ref_logits), axis=-1)
    cur_top = np.argmax(np.asarray(cur_logits), axis=-1)
    if ref_top.shape != cur_top.shape:
        raise ValueError(f"Cannot compare top1 for mismatch: {ref_top.shape} vs {cur_top.shape}")
    return float(np.mean(ref_top != cur_top))
