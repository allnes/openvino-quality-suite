from __future__ import annotations


def maybe_tqdm(iterable, enabled: bool = True):
    if not enabled:
        return iterable
    from tqdm import tqdm

    return tqdm(iterable)
