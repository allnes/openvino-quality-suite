from __future__ import annotations


def metric_delta(
    baseline: dict, current: dict, keys: list[str]
) -> dict[str, dict[str, float | None]]:
    out = {}
    for key in keys:
        base = baseline.get(key)
        cur = current.get(key)
        out[key] = {
            "baseline": base,
            "current": cur,
            "delta": None if base is None or cur is None else cur - base,
        }
    return out
