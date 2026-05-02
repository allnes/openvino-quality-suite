from __future__ import annotations

import numpy as np

from oviqs.adapters.errors import OptionalDependencyError
from oviqs.adapters.runners.base import BaseGenerationRunner, BaseLogitsRunner
from oviqs.ports.runners import RunnerCapabilities


class HFReferenceRunner(BaseLogitsRunner, BaseGenerationRunner):
    name = "hf-reference"
    capabilities = RunnerCapabilities(
        exposes_logits=True,
        supports_generation=True,
        supports_dynamic_shapes=True,
        supported_devices=("cpu", "cuda", "mps"),
    )

    def __init__(self, model_id_or_path: str, device: str = "cpu", dtype: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise OptionalDependencyError("transformers/torch", "openvino") from exc

        self.torch = torch
        self.model_id_or_path = model_id_or_path
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
            model_id_or_path, use_fast=True
        )
        torch_dtype = (
            getattr(torch, dtype) if dtype in {"float16", "bfloat16", "float32"} else "auto"
        )
        self.model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            model_id_or_path, torch_dtype=torch_dtype
        )
        self.model.to(device)  # type: ignore[arg-type]
        self.model.eval()

    def run_info(self) -> dict:
        return {
            "backend": "hf",
            "model_id": self.model_id_or_path,
            "device": self.device,
            "dtype": self.dtype,
        }

    def encode(self, text: str):
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=True)

    def forward_logits(self, input_ids, attention_mask=None):
        torch = self.torch
        if not hasattr(input_ids, "to"):
            input_ids = torch.tensor(input_ids)
        input_ids = input_ids.to(self.device)
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            if not hasattr(attention_mask, "to"):
                attention_mask = torch.tensor(attention_mask)
            kwargs["attention_mask"] = attention_mask.to(self.device)
        with torch.no_grad():
            out = self.model(**kwargs, use_cache=False)
        return out.logits.detach().cpu().float().numpy()

    def generate(self, prompt: str, **generation_kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out_ids = self.model.generate(**inputs, **generation_kwargs)
        return self.tokenizer.decode(
            np.asarray(out_ids[0].detach().cpu()), skip_special_tokens=False
        )
