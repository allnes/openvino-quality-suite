from __future__ import annotations

from typing import Any

from oviqs.domain.gates import evaluate_gates


class DomainGateEvaluator:
    def evaluate(
        self,
        report: dict[str, Any],
        gate_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not gate_payload:
            return {}
        if "sections" in gate_payload and "overall_status" in gate_payload:
            return gate_payload
        return evaluate_gates(report, gate_payload)


__all__ = ["DomainGateEvaluator"]
