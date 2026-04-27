from __future__ import annotations

from collections import Counter


def ngram_repetition_rate(text: str, n: int = 3) -> dict[str, float | int]:
    tokens = text.split()
    if len(tokens) < n:
        return {"n": n, "total_ngrams": 0, "repetition_rate": 0.0, "unique_ngram_ratio": 1.0}
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "n": n,
        "total_ngrams": len(ngrams),
        "repetition_rate": repeated / len(ngrams),
        "unique_ngram_ratio": len(counts) / len(ngrams),
    }


def json_validity(text: str) -> dict[str, bool | str | None]:
    import json

    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return {"json_valid": False, "error": str(exc)}
    return {"json_valid": True, "error": None}
