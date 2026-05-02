import numpy as np

from oviqs.adapters.runners.ov_runtime import OVRuntimeLogitsRunner, _resolve_model_and_tokenizer


def test_resolve_openvino_runtime_directory_prefers_openvino_model_xml(tmp_path):
    (tmp_path / "z.xml").write_text("<xml/>", encoding="utf-8")
    preferred = tmp_path / "openvino_model.xml"
    preferred.write_text("<xml/>", encoding="utf-8")

    model_xml, tokenizer_dir = _resolve_model_and_tokenizer(str(tmp_path), None)

    assert model_xml == preferred
    assert tokenizer_dir == tmp_path


def test_resolve_openvino_runtime_directory_prefers_language_model_xml(tmp_path):
    (tmp_path / "openvino_detokenizer.xml").write_text("<xml/>", encoding="utf-8")
    preferred = tmp_path / "openvino_language_model.xml"
    preferred.write_text("<xml/>", encoding="utf-8")

    model_xml, tokenizer_dir = _resolve_model_and_tokenizer(str(tmp_path), None)

    assert model_xml == preferred
    assert tokenizer_dir == tmp_path


def test_build_inputs_repeats_position_ids_for_batch():
    runner = object.__new__(OVRuntimeLogitsRunner)
    runner.inputs = {
        "input_ids": object(),
        "attention_mask": object(),
        "position_ids": object(),
        "cache_position": object(),
        "q_length": object(),
        "q_offset": object(),
    }

    feeds = runner._build_inputs(
        input_ids=np.array([[1, 2, 3], [4, 5, 0]]),
        attention_mask=np.array([[1, 1, 1], [1, 1, 0]]),
    )

    assert feeds["position_ids"].shape == (2, 3)
    assert feeds["position_ids"].tolist() == [[0, 1, 2], [0, 1, 2]]
    assert feeds["cache_position"].tolist() == [0, 1, 2]
    assert feeds["q_length"].tolist() == [3]
    assert feeds["q_offset"].tolist() == [0]


def test_build_inputs_supports_embedding_model_feeds():
    runner = object.__new__(OVRuntimeLogitsRunner)
    runner.inputs = {
        "inputs_embeds": object(),
        "attention_mask": object(),
        "beam_idx": object(),
    }

    feeds = runner._build_inputs(
        input_ids=np.array([[1, 2, 3], [4, 5, 0]]),
        attention_mask=np.array([[1, 1, 1], [1, 1, 0]]),
        inputs_embeds=np.ones((2, 3, 8), dtype=np.float32),
    )

    assert feeds["inputs_embeds"].shape == (2, 3, 8)
    assert feeds["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert feeds["beam_idx"].tolist() == [0, 1]


def test_build_cached_step_inputs_uses_decode_position_and_full_attention_prefix():
    runner = object.__new__(OVRuntimeLogitsRunner)
    runner.inputs = {
        "input_ids": object(),
        "attention_mask": object(),
        "position_ids": object(),
        "beam_idx": object(),
        "cache_position": object(),
        "q_length": object(),
        "q_offset": object(),
    }

    feeds = runner._build_cached_step_inputs(
        input_ids=np.array([[7]]),
        position=5,
        attention_mask=np.ones((1, 6), dtype=np.int64),
    )

    assert feeds["input_ids"].tolist() == [[7]]
    assert feeds["attention_mask"].shape == (1, 6)
    assert feeds["position_ids"].tolist() == [[5]]
    assert feeds["beam_idx"].tolist() == [0]
    assert feeds["cache_position"].tolist() == [5]
    assert feeds["q_length"].tolist() == [1]
    assert feeds["q_offset"].tolist() == [5]
