import numpy as np
import pytest

from oviqs.domain.metrics.distribution_drift import (
    aggregate_drift,
    distribution_drift,
    topk_overlap,
)


def test_identical_logits_have_zero_drift():
    logits = np.array([[[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]]], dtype=np.float32)
    agg = aggregate_drift(distribution_drift(logits, logits))
    assert agg["mean_kl"] == pytest.approx(0.0, abs=1e-7)
    assert agg["mean_js"] == pytest.approx(0.0, abs=1e-7)
    assert agg["mean_entropy_drift"] == pytest.approx(0.0, abs=1e-7)
    assert agg["mean_logit_cosine"] == pytest.approx(1.0)


def test_topk_overlap_identical_is_one():
    logits = np.array([[[1.0, 4.0, 2.0, 3.0]]])
    assert topk_overlap(logits, logits, k=2) == pytest.approx(1.0)
