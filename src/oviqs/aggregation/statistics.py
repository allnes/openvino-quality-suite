from __future__ import annotations

import numpy as np


def mean_with_count(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "count": 0}
    return {"mean": float(np.mean(values)), "count": len(values)}
