from __future__ import annotations

from oviqs.core.sample import EvalSample


def validate_samples(samples: list[EvalSample]) -> list[EvalSample]:
    return samples


__all__ = ["validate_samples"]
