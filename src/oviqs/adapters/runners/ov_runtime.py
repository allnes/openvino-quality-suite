from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oviqs.adapters.errors import OptionalDependencyError
from oviqs.adapters.runners.base import BaseLogitsRunner
from oviqs.ports.runners import RunnerCapabilities


class OVRuntimeLogitsRunner(BaseLogitsRunner):
    name = "openvino-runtime"
    capabilities = RunnerCapabilities(
        exposes_logits=True,
        supports_stateful_kv=True,
        supports_dynamic_shapes=True,
        supported_devices=("CPU", "GPU", "NPU", "AUTO", "HETERO"),
    )

    def __init__(
        self,
        model_xml: str,
        device: str = "CPU",
        config: dict | None = None,
        tokenizer_dir: str | None = None,
    ) -> None:
        try:
            import openvino as ov
        except ImportError as exc:
            raise OptionalDependencyError("openvino", "openvino") from exc

        resolved_model_xml, resolved_tokenizer_dir = _resolve_model_and_tokenizer(
            model_xml, tokenizer_dir
        )
        self.model_xml = str(resolved_model_xml)
        self.tokenizer_dir = str(resolved_tokenizer_dir) if resolved_tokenizer_dir else None
        self.model_dir = resolved_model_xml.parent if resolved_model_xml.is_file() else None
        self.device = device
        self.config = config or {}
        self.tokenizer = _load_tokenizer(self.tokenizer_dir) if self.tokenizer_dir else None
        self.core = ov.Core()
        self.model = self.core.read_model(self.model_xml)
        self.inputs = {inp.get_any_name(): inp for inp in self.model.inputs}
        self.outputs = {out.get_any_name(): out for out in self.model.outputs}
        self.embedding_model_xml = _resolve_text_embeddings_model(resolved_model_xml)
        self.embedding_compiled = None
        self.embedding_request = None
        self.compiled = None
        self.request = None
        self._compiled_shape_key: tuple[tuple[str, tuple[int, ...]], ...] | None = None

    def run_info(self):
        return {
            "backend": "openvino-runtime",
            "model_xml": self.model_xml,
            "tokenizer_dir": self.tokenizer_dir,
            "embedding_model_xml": (
                str(self.embedding_model_xml) if self.embedding_model_xml else None
            ),
            "device": self.device,
            "config": self.config,
        }

    def encode(self, text: str):
        if self.tokenizer is None:
            raise RuntimeError(
                "OVRuntimeLogitsRunner has no tokenizer. Pass an exported model directory "
                "or tokenizer_dir."
            )
        return self.tokenizer(text, return_tensors="np", add_special_tokens=True)

    def _build_inputs(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray | None = None,
        inputs_embeds: np.ndarray | None = None,
    ):
        feeds = {}
        if "input_ids" in self.inputs:
            feeds["input_ids"] = input_ids.astype(np.int64)
        elif "inputs_embeds" in self.inputs:
            if inputs_embeds is None:
                raise RuntimeError(
                    "OpenVINO model expects inputs_embeds, but no text embeddings model "
                    "was found next to the language model."
                )
            feeds["inputs_embeds"] = inputs_embeds.astype(np.float32)
        else:
            first_name = next(iter(self.inputs.keys()))
            feeds[first_name] = input_ids.astype(np.int64)
        if attention_mask is not None and "attention_mask" in self.inputs:
            feeds["attention_mask"] = attention_mask.astype(np.int64)
        if "beam_idx" in self.inputs:
            feeds["beam_idx"] = np.arange(input_ids.shape[0], dtype=np.int32)
        if "position_ids" in self.inputs:
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            feeds["position_ids"] = np.broadcast_to(
                np.arange(seq_len, dtype=np.int64)[None, :],
                (batch_size, seq_len),
            )
        if "cache_position" in self.inputs:
            feeds["cache_position"] = np.arange(input_ids.shape[1], dtype=np.int64)
        if "q_length" in self.inputs:
            feeds["q_length"] = np.asarray([input_ids.shape[1]], dtype=np.int64)
        if "q_offset" in self.inputs:
            feeds["q_offset"] = np.asarray([0], dtype=np.int64)
        return feeds

    def forward_logits(self, input_ids: np.ndarray, attention_mask: np.ndarray | None = None):
        input_ids = np.asarray(input_ids)
        attention_mask = None if attention_mask is None else np.asarray(attention_mask)
        inputs_embeds = self._embed_inputs(input_ids) if self._expects_inputs_embeds() else None
        self._ensure_compiled(input_ids, attention_mask, inputs_embeds)
        feeds = self._build_inputs(input_ids, attention_mask, inputs_embeds)
        if self.request is None:
            raise RuntimeError("OpenVINO infer request was not initialized")
        self._reset_state_if_needed()
        outputs = self.request.infer(feeds)
        for value in outputs.values():
            arr = np.asarray(value)
            if arr.ndim == 3:
                return arr.astype(np.float32)
        raise RuntimeError("Cannot find logits output with rank 3 [B, T, V].")

    def supports_stateful_cache(self) -> bool:
        """Return whether the compiled OpenVINO model exposes inference state."""

        dummy_ids = np.zeros((1, 1), dtype=np.int64)
        dummy_mask = np.ones_like(dummy_ids, dtype=np.int64)
        try:
            dummy_embeds = self._embed_inputs(dummy_ids) if self._expects_inputs_embeds() else None
            self._ensure_compiled(dummy_ids, dummy_mask, dummy_embeds)
        except Exception:
            return False
        return bool(self._query_state())

    def forward_logits_cached_decode(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run one-token-at-a-time decode while preserving OpenVINO KV-cache state.

        The returned logits have shape `[B, T-1, V]` and align with
        `forward_logits(input_ids)[:, :-1, :]`, i.e. the positions used for next-token
        likelihood and KV-cache drift checks.
        """

        input_ids = np.asarray(input_ids)
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {input_ids.shape}")
        if input_ids.shape[0] != 1:
            raise ValueError("cached decode currently supports batch size 1")
        if input_ids.shape[1] < 2:
            raise ValueError("Need at least two tokens for cached decode comparison")
        attention_mask = (
            np.ones_like(input_ids, dtype=np.int64)
            if attention_mask is None
            else np.asarray(attention_mask, dtype=np.int64)
        )
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape mismatch: {attention_mask.shape} vs {input_ids.shape}"
            )

        first_ids = input_ids[:, :1]
        first_embeds = self._embed_inputs(first_ids) if self._expects_inputs_embeds() else None
        self._ensure_compiled(first_ids, attention_mask[:, :1], first_embeds)
        states = self._query_state()
        if not states:
            raise RuntimeError("OpenVINO model does not expose state for cached decode")
        for state in states:
            state.reset()

        outputs: list[np.ndarray] = []
        for pos in range(input_ids.shape[1] - 1):
            step_ids = input_ids[:, pos : pos + 1]
            step_embeds = self._embed_inputs(step_ids) if self._expects_inputs_embeds() else None
            feeds = self._build_cached_step_inputs(
                step_ids,
                position=pos,
                attention_mask=attention_mask[:, : pos + 1],
                inputs_embeds=step_embeds,
            )
            if self.request is None:
                raise RuntimeError("OpenVINO infer request was not initialized")
            result = self.request.infer(feeds)
            outputs.append(self._extract_logits(result)[:, -1:, :])

        return np.concatenate(outputs, axis=1).astype(np.float32)

    def _ensure_compiled(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray | None = None,
        inputs_embeds: np.ndarray | None = None,
    ) -> None:
        shapes = (
            {}
            if self._requires_dynamic_compile()
            else self._input_shapes(input_ids, attention_mask, inputs_embeds)
        )
        shape_key = tuple(sorted((name, tuple(shape)) for name, shape in shapes.items()))
        if self.compiled is not None and self._compiled_shape_key == shape_key:
            return

        model = self.core.read_model(self.model_xml)
        if shapes:
            model.reshape(shapes)
        try:
            compiled = self.core.compile_model(model, self.device, self.config)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to compile OpenVINO model for device={self.device}, shapes={shapes}"
            ) from exc
        self.compiled = compiled
        self.request = compiled.create_infer_request()
        self._compiled_shape_key = shape_key

    def _input_shapes(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray | None = None,
        inputs_embeds: np.ndarray | None = None,
    ) -> dict[str, list[int]]:
        shapes: dict[str, list[int]] = {}
        if "input_ids" in self.inputs:
            shapes["input_ids"] = list(input_ids.shape)
        elif "inputs_embeds" in self.inputs:
            if inputs_embeds is None:
                raise RuntimeError("Cannot reshape inputs_embeds model without embeddings.")
            shapes["inputs_embeds"] = list(inputs_embeds.shape)
        else:
            fallback_name = next(iter(self.inputs.keys()))
            shapes[fallback_name] = list(input_ids.shape)
        if "attention_mask" in self.inputs:
            shapes["attention_mask"] = (
                list(attention_mask.shape) if attention_mask is not None else list(input_ids.shape)
            )
        if "beam_idx" in self.inputs:
            shapes["beam_idx"] = [input_ids.shape[0]]
        if "position_ids" in self.inputs:
            shapes["position_ids"] = list(input_ids.shape)
        if "cache_position" in self.inputs:
            shapes["cache_position"] = [input_ids.shape[1]]
        if "q_length" in self.inputs:
            shapes["q_length"] = [1]
        if "q_offset" in self.inputs:
            shapes["q_offset"] = [1]
        return shapes

    def _expects_inputs_embeds(self) -> bool:
        return "inputs_embeds" in self.inputs and "input_ids" not in self.inputs

    def _requires_dynamic_compile(self) -> bool:
        return self._expects_inputs_embeds() or "beam_idx" in self.inputs

    def _embed_inputs(self, input_ids: np.ndarray) -> np.ndarray | None:
        if self.embedding_model_xml is None:
            return None
        if self.embedding_compiled is None:
            embedding_model = self.core.read_model(str(self.embedding_model_xml))
            embedding_compiled = self.core.compile_model(
                embedding_model,
                self.device,
                self.config,
            )
            self.embedding_compiled = embedding_compiled
            self.embedding_request = embedding_compiled.create_infer_request()
        if self.embedding_request is None or self.embedding_compiled is None:
            raise RuntimeError("OpenVINO text embeddings infer request was not initialized")
        input_name = self.embedding_compiled.inputs[0].get_any_name()
        outputs = self.embedding_request.infer({input_name: input_ids.astype(np.int64)})
        for value in outputs.values():
            arr = np.asarray(value)
            if arr.ndim == 3:
                return arr.astype(np.float32)
        raise RuntimeError("Cannot find text embeddings output with rank 3 [B, T, H].")

    def _build_cached_step_inputs(
        self,
        input_ids: np.ndarray,
        *,
        position: int,
        attention_mask: np.ndarray,
        inputs_embeds: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        feeds: dict[str, np.ndarray] = {}
        batch_size = input_ids.shape[0]
        if "input_ids" in self.inputs:
            feeds["input_ids"] = input_ids.astype(np.int64)
        elif "inputs_embeds" in self.inputs:
            if inputs_embeds is None:
                raise RuntimeError("OpenVINO model expects inputs_embeds for cached decode")
            feeds["inputs_embeds"] = inputs_embeds.astype(np.float32)
        else:
            first_name = next(iter(self.inputs.keys()))
            feeds[first_name] = input_ids.astype(np.int64)
        if "attention_mask" in self.inputs:
            feeds["attention_mask"] = attention_mask.astype(np.int64)
        if "beam_idx" in self.inputs:
            feeds["beam_idx"] = np.arange(batch_size, dtype=np.int32)
        if "position_ids" in self.inputs:
            feeds["position_ids"] = np.full((batch_size, 1), position, dtype=np.int64)
        if "cache_position" in self.inputs:
            feeds["cache_position"] = np.asarray([position], dtype=np.int64)
        if "q_length" in self.inputs:
            feeds["q_length"] = np.asarray([1], dtype=np.int64)
        if "q_offset" in self.inputs:
            feeds["q_offset"] = np.asarray([position], dtype=np.int64)
        return feeds

    def _extract_logits(self, outputs) -> np.ndarray:
        for value in outputs.values():
            arr = np.asarray(value)
            if arr.ndim == 3:
                return arr.astype(np.float32)
        raise RuntimeError("Cannot find logits output with rank 3 [B, T, V].")

    def _reset_state_if_needed(self) -> None:
        for state in self._query_state():
            state.reset()

    def _query_state(self):
        if self.request is None:
            return []
        query_state = getattr(self.request, "query_state", None)
        if query_state is None:
            return []
        return list(query_state())


def _resolve_model_and_tokenizer(
    model_xml_or_dir: str,
    tokenizer_dir: str | None,
) -> tuple[Path, Path | None]:
    path = Path(model_xml_or_dir)
    if path.is_dir():
        xml_files = sorted(path.glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No OpenVINO .xml model found in {path}")
        if len(xml_files) > 1:
            preferred_names = ("openvino_model.xml", "openvino_language_model.xml")
            preferred = [
                item for name in preferred_names for item in xml_files if item.name == name
            ]
            model_xml = preferred[0] if preferred else xml_files[0]
        else:
            model_xml = xml_files[0]
        return model_xml, Path(tokenizer_dir) if tokenizer_dir else path
    return path, Path(tokenizer_dir) if tokenizer_dir else None


def _resolve_text_embeddings_model(model_xml: Path) -> Path | None:
    candidate = model_xml.parent / "openvino_text_embeddings_model.xml"
    if model_xml.name == "openvino_language_model.xml" and candidate.exists():
        return candidate
    return None


def _load_tokenizer(tokenizer_dir: str):
    try:
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
    except ImportError as exc:
        raise OptionalDependencyError("transformers", "openvino") from exc
    try:
        return AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)  # nosec B615
    except ValueError:
        tokenizer_path = Path(tokenizer_dir) / "tokenizer.json"
        tokenizer_config_path = Path(tokenizer_dir) / "tokenizer_config.json"
        if not tokenizer_path.exists():
            raise
        kwargs = {"tokenizer_file": str(tokenizer_path)}
        if tokenizer_config_path.exists():
            config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
            for key in ("bos_token", "eos_token", "unk_token", "pad_token"):
                if config.get(key) is not None:
                    kwargs[key] = config[key]
        return PreTrainedTokenizerFast(**kwargs)
