from oviqs.metrics.rag import (
    citation_metrics,
    context_precision,
    context_recall,
    distractor_ratio,
    evidence_coverage,
    rule_based_faithfulness,
    supported_claim_ratio_placeholder,
)


def test_evidence_coverage():
    metrics = evidence_coverage(["April 16", "VX-913"], ["The deadline is April 16."])
    assert metrics["matched_evidence"] == 1
    assert metrics["evidence_coverage"] == 0.5


def test_supported_claim_placeholder_is_not_fake_score():
    metrics = supported_claim_ratio_placeholder()
    assert metrics["supported_claim_ratio"] is None
    assert metrics["warnings"]


def test_context_and_citation_metrics():
    contexts = ["The deadline is April 16.", "Unrelated text"]
    assert context_precision(["April 16"], contexts)["context_precision"] == 0.5
    assert context_recall(["April 16", "VX-913"], contexts)["context_recall"] == 0.5
    citations = citation_metrics(["doc1", "doc2"], ["doc1", "doc3"])
    assert citations["citation_precision"] == 0.5
    assert citations["citation_recall"] == 0.5


def test_rule_based_faithfulness_and_distractors():
    metrics = rule_based_faithfulness(
        "The deadline is April 16.",
        ["The deadline is April 16."],
        ["April 16"],
    )
    assert metrics["faithfulness"] == 1.0
    assert distractor_ratio(["a", "b", "c"], [0, 2])["distractor_ratio"] == 1 / 3
