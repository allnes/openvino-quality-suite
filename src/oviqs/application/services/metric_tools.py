from __future__ import annotations

import json
from typing import Any

from oviqs.domain.metrics.long_context import (
    context_gain,
    degradation_slope,
    distractor_sensitivity,
    lost_in_middle_score_from_ppl,
)


def lost_in_middle_from_json(position_ppl_json: str) -> dict[str, Any]:
    values = json.loads(position_ppl_json)
    return {"lost_in_middle_score": lost_in_middle_score_from_ppl(values)}


__all__ = [
    "context_gain",
    "degradation_slope",
    "distractor_sensitivity",
    "lost_in_middle_from_json",
]
