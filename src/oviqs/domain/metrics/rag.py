from __future__ import annotations


def evidence_coverage(
    expected_evidence: list[str], retrieved_contexts: list[str]
) -> dict[str, float | int]:
    haystack = "\n".join(retrieved_contexts).lower()
    matched = sum(1 for item in expected_evidence if item.lower() in haystack)
    total = len(expected_evidence)
    return {
        "expected_evidence": total,
        "matched_evidence": matched,
        "evidence_coverage": matched / max(total, 1),
    }


def context_precision(
    expected_evidence: list[str],
    retrieved_contexts: list[str],
) -> dict[str, float | int]:
    """Fraction of retrieved contexts that contain expected evidence."""

    if not retrieved_contexts:
        return {"retrieved_contexts": 0, "relevant_contexts": 0, "context_precision": 0.0}
    expected = [item.lower() for item in expected_evidence]
    relevant = 0
    for context in retrieved_contexts:
        lower = context.lower()
        if any(item in lower for item in expected):
            relevant += 1
    return {
        "retrieved_contexts": len(retrieved_contexts),
        "relevant_contexts": relevant,
        "context_precision": relevant / len(retrieved_contexts),
    }


def context_recall(
    expected_evidence: list[str],
    retrieved_contexts: list[str],
) -> dict[str, float | int]:
    """Fraction of expected evidence strings covered by retrieved contexts."""

    coverage = evidence_coverage(expected_evidence, retrieved_contexts)
    return {
        "expected_evidence": coverage["expected_evidence"],
        "matched_evidence": coverage["matched_evidence"],
        "context_recall": coverage["evidence_coverage"],
    }


def citation_metrics(
    expected_citations: list[str],
    actual_citations: list[str],
) -> dict[str, float | int]:
    """Compute exact-match citation precision and recall."""

    expected = set(expected_citations)
    actual = set(actual_citations)
    matched = len(expected & actual)
    return {
        "expected_citations": len(expected),
        "actual_citations": len(actual),
        "matched_citations": matched,
        "citation_precision": matched / max(len(actual), 1),
        "citation_recall": matched / max(len(expected), 1),
    }


def rule_based_faithfulness(
    answer: str,
    retrieved_contexts: list[str],
    claims: list[str],
) -> dict[str, float | int]:
    """Estimate faithfulness with literal claim coverage.

    This is intentionally conservative and does not replace a judge model. It only marks
    a claim supported when its text appears in the retrieved contexts or answer context.
    """

    haystack = "\n".join([answer, *retrieved_contexts]).lower()
    supported = sum(1 for claim in claims if claim.lower() in haystack)
    return {
        "claims": len(claims),
        "supported_claims": supported,
        "faithfulness": supported / max(len(claims), 1),
    }


def distractor_ratio(
    retrieved_contexts: list[str],
    relevant_context_indices: list[int],
) -> dict[str, float | int]:
    """Return the ratio of retrieved contexts not marked relevant."""

    total = len(retrieved_contexts)
    relevant = len(set(relevant_context_indices))
    distractors = max(total - relevant, 0)
    return {
        "retrieved_contexts": total,
        "relevant_contexts": relevant,
        "distractor_contexts": distractors,
        "distractor_ratio": distractors / max(total, 1),
    }


def supported_claim_ratio_placeholder() -> dict[str, None | list[str]]:
    return {
        "supported_claim_ratio": None,
        "warnings": ["supported_claim_ratio requires judge_model or external scorer"],
    }
