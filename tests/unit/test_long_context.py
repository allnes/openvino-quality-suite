import pytest

from oviqs.metrics.long_context import (
    authoritative_margin,
    conflict_entropy,
    conflict_sensitivity,
    context_gain,
    degradation_slope,
    distractor_sensitivity,
)


def test_context_gain():
    assert context_gain({"0k": 2.0, "4k": 1.5}) == {"4k": 0.5}


def test_degradation_slope():
    assert degradation_slope({4: -1.0, 8: -2.0}) == pytest.approx(-1.0)


def test_distractor_sensitivity():
    assert distractor_sensitivity(1.0, 1.4) == pytest.approx(0.4)


def test_conflict_metrics():
    assert conflict_sensitivity(1.0, 1.7) == pytest.approx(0.7)
    assert authoritative_margin({"auth": -0.1, "bad": -1.2}, "auth") == pytest.approx(1.1)
    assert conflict_entropy({"a": -1.0, "b": -1.0}) == pytest.approx(0.693147, rel=1e-5)
