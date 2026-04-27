import pytest

pytestmark = pytest.mark.integration


def test_small_hf_runner_placeholder():
    pytest.skip("Requires an explicit tiny HF model fixture.")
