import numpy as np
import pytest

from oviqs.domain.metrics.serving import generation_prefix_divergence, kv_cache_drift


def test_generation_prefix_divergence():
    metrics = generation_prefix_divergence("a b c", "a b d")
    assert metrics["prefix_match_tokens"] == 2
    assert metrics["prefix_divergence_rate"] == pytest.approx(1 / 3)


def test_kv_cache_drift_identical_logits():
    logits = np.array([[[1.0, 2.0], [3.0, 1.0]]], dtype=np.float32)
    metrics = kv_cache_drift(logits, logits)
    assert metrics["mean_kl"] == 0.0
    assert metrics["top1_changed_rate"] == 0.0
