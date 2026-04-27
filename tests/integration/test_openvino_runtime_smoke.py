import pytest

pytestmark = pytest.mark.openvino


def test_openvino_runtime_smoke_placeholder():
    pytest.skip("Requires an exported OpenVINO IR model path.")
