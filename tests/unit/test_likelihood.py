import numpy as np
import pytest

from oviqs.adapters.runners.dummy import DummyLogitsRunner
from oviqs.domain.metrics.likelihood import (
    nll_ppl_from_logits,
    sliding_window_ppl,
    token_logprobs_from_logits,
)


def test_uniform_distribution_ppl_matches_vocab_size():
    logits = np.zeros((1, 4, 5), dtype=np.float32)
    input_ids = np.array([[0, 1, 2, 3]])
    metrics = nll_ppl_from_logits(logits, input_ids)
    assert metrics["num_tokens"] == 3
    assert metrics["perplexity"] == pytest.approx(5.0)


def test_target_mask_selects_only_requested_tokens():
    logits = np.zeros((1, 4, 5), dtype=np.float32)
    input_ids = np.array([[0, 1, 2, 3]])
    target_mask = np.array([[0, 0, 1, 0]])
    logprobs, mask = token_logprobs_from_logits(logits, input_ids, target_mask=target_mask)
    assert logprobs.shape == (1, 3)
    assert mask.tolist() == [[False, True, False]]


def test_biased_logits_have_low_ppl():
    runner = DummyLogitsRunner(vocab_size=8, correct_bias=10.0)
    ids = np.array([[1, 2, 3, 4]])
    metrics = nll_ppl_from_logits(runner.forward_logits(ids), ids)
    assert metrics["perplexity"] < 1.01


def test_sliding_window_does_not_double_count():
    runner = DummyLogitsRunner(vocab_size=8, correct_bias=10.0)
    ids = np.array([[1, 2, 3, 4, 5, 6]])
    metrics = sliding_window_ppl(runner, ids, window_size=4, stride=2)
    assert metrics["num_tokens"] == 5
    assert [item["absolute_pos"] for item in metrics["per_token"]] == [1, 2, 3, 4, 5]
