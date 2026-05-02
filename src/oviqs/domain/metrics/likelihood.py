from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import log_softmax


def _as_array(value: Any, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def token_logprobs_from_logits(
    logits,
    input_ids,
    attention_mask=None,
    target_mask=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log p(x_t | x_<t) and a boolean mask.

    `logits[:, :-1]` predicts `input_ids[:, 1:]`. Padding and optional target masks are
    applied after the one-token causal shift.
    """

    logits_arr = _as_array(logits, np.float32)
    input_arr = _as_array(input_ids)
    if logits_arr.ndim != 3:
        raise ValueError(f"logits must have shape [B, T, V], got {logits_arr.shape}")
    if input_arr.ndim != 2:
        raise ValueError(f"input_ids must have shape [B, T], got {input_arr.shape}")
    if logits_arr.shape[:2] != input_arr.shape:
        raise ValueError(
            f"logits/input_ids shape mismatch: {logits_arr.shape} vs {input_arr.shape}"
        )
    if logits_arr.shape[1] < 2:
        raise ValueError("Need at least two tokens to compute causal likelihood")

    pred_logits = logits_arr[:, :-1, :]
    target_ids = input_arr[:, 1:].astype(np.int64)
    vocab_size = pred_logits.shape[-1]
    if np.any(target_ids < 0) or np.any(target_ids >= vocab_size):
        raise ValueError("target token id outside logits vocabulary")

    log_probs = log_softmax(pred_logits, axis=-1)
    target_logprobs = np.take_along_axis(log_probs, target_ids[..., None], axis=-1).squeeze(-1)

    if attention_mask is not None:
        mask = _as_array(attention_mask)[:, 1:].astype(bool)
    else:
        mask = np.ones_like(target_ids, dtype=bool)
    if target_mask is not None:
        mask = mask & _as_array(target_mask)[:, 1:].astype(bool)

    return target_logprobs.astype(np.float32), mask


def nll_ppl_from_logits(
    logits, input_ids, attention_mask=None, target_mask=None
) -> dict[str, float | int]:
    """Compute causal LM mean NLL and perplexity.

    NLL is `-mean log p(x_t | x_<t)`. PPL is `exp(NLL)`. Aggregate by averaging NLL,
    never by averaging per-token perplexities.
    """

    target_logprobs, mask = token_logprobs_from_logits(
        logits,
        input_ids,
        attention_mask=attention_mask,
        target_mask=target_mask,
    )
    selected = target_logprobs[mask]
    if selected.size == 0:
        raise ValueError("No tokens selected for NLL/PPL calculation")

    nll = float(-np.mean(selected))
    return {
        "nll": nll,
        "perplexity": float(np.exp(nll)),
        "mean_log_prob": float(np.mean(selected)),
        "num_tokens": int(selected.size),
    }


def sliding_window_ppl(
    runner,
    input_ids,
    attention_mask=None,
    window_size: int = 4096,
    stride: int = 1024,
) -> dict[str, Any]:
    """Compute sliding-window PPL without double-counting overlapping context tokens."""

    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if stride < 1:
        raise ValueError("stride must be positive")

    input_arr = _as_array(input_ids)
    mask_arr = _as_array(attention_mask) if attention_mask is not None else None
    total_nll_sum = 0.0
    total_tokens = 0
    per_token: list[dict[str, float | int]] = []
    seq_len = input_arr.shape[1]

    for start in range(0, max(seq_len - 1, 0), stride):
        end = min(start + window_size, seq_len)
        window_ids = input_arr[:, start:end]
        window_mask = mask_arr[:, start:end] if mask_arr is not None else None
        if window_ids.shape[1] < 2:
            continue

        logits = runner.forward_logits(window_ids, window_mask)
        token_logprobs, mask = token_logprobs_from_logits(logits, window_ids, window_mask)

        score_mask = np.zeros_like(mask, dtype=bool)
        keep_from = 0 if start == 0 else max(0, token_logprobs.shape[1] - stride)
        score_mask[:, keep_from:] = mask[:, keep_from:]
        selected = token_logprobs[score_mask]
        total_nll_sum += float(-np.sum(selected))
        total_tokens += int(selected.size)

        for local_idx, lp in enumerate(token_logprobs[0]):
            if score_mask[0, local_idx]:
                absolute_pos = start + local_idx + 1
                per_token.append(
                    {"absolute_pos": int(absolute_pos), "log_prob": float(lp), "nll": float(-lp)}
                )

        if end == seq_len:
            break

    if total_tokens == 0:
        raise ValueError("No tokens selected for sliding-window PPL calculation")
    mean_nll = total_nll_sum / total_tokens
    return {
        "nll": mean_nll,
        "perplexity": float(np.exp(mean_nll)),
        "num_tokens": total_tokens,
        "per_token": per_token,
    }
