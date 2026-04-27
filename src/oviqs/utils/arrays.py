from __future__ import annotations

import numpy as np


def align_last_dim(left, right):
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    vocab = min(left_arr.shape[-1], right_arr.shape[-1])
    return left_arr[..., :vocab], right_arr[..., :vocab]
