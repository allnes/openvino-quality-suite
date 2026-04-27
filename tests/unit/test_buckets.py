import pytest

from oviqs.aggregation.buckets import (
    aggregate_position_bucketed_ppl,
    effective_context_bucket,
    relative_position_bucket,
    sample_length_bucket,
)
from oviqs.metrics.long_context import lost_in_middle_score_from_ppl


def test_length_buckets():
    assert sample_length_bucket(4096) == "0_4k"
    assert sample_length_bucket(4097) == "4_16k"
    assert effective_context_bucket(70000) == "ctx_64k_plus"


def test_position_buckets_and_aggregation():
    assert relative_position_bucket(0, 100) == "0_10"
    assert relative_position_bucket(99, 100) == "90_100"
    agg = aggregate_position_bucketed_ppl(
        [{"absolute_pos": 0, "nll": 1.0}, {"absolute_pos": 99, "nll": 2.0}],
        seq_len=100,
    )
    assert agg["0_10"]["tokens"] == 1
    assert agg["90_100"]["tokens"] == 1


def test_lost_in_middle_formula():
    score = lost_in_middle_score_from_ppl(
        {"0_10": 5.0, "30_50": 10.0, "50_70": 10.0, "90_100": 5.0}
    )
    assert score == pytest.approx(1.0)
