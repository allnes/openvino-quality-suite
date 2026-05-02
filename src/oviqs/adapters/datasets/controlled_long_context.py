from __future__ import annotations

from oviqs.domain.samples import EvalSample


def build_needle_sample(
    sample_id: str, context: str, prompt: str, target: str, position: str
) -> EvalSample:
    return EvalSample(
        id=sample_id,
        task_type="controlled_context",
        context=context,
        prompt=prompt,
        target=target,
        metadata={"fact_position": position},
    )
