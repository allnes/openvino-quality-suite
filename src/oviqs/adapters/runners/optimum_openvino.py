from __future__ import annotations

from oviqs.adapters.errors import OptionalDependencyError
from oviqs.adapters.runners.base import BaseGenerationRunner, BaseLogitsRunner
from oviqs.ports.runners import RunnerCapabilities


class OptimumOVLogitsRunner(BaseLogitsRunner, BaseGenerationRunner):
    name = "optimum-openvino"
    capabilities = RunnerCapabilities(
        exposes_logits=True,
        supports_generation=True,
        supports_dynamic_shapes=True,
        supported_devices=("CPU", "GPU", "NPU", "AUTO"),
    )

    def __init__(self, model_dir: str, device: str = "CPU") -> None:
        try:
            import torch
            from optimum.intel import OVModelForCausalLM
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise OptionalDependencyError("optimum-intel/transformers/torch", "openvino") from exc

        self.torch = torch
        self.model_dir = model_dir
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)  # nosec B615
        self.model = OVModelForCausalLM.from_pretrained(model_dir, device=device)
        self.model.eval()

    def run_info(self):
        return {"backend": "optimum-openvino", "model_dir": self.model_dir, "device": self.device}

    def encode(self, text: str):
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=True)

    def forward_logits(self, input_ids, attention_mask=None):
        torch = self.torch
        if not hasattr(input_ids, "detach"):
            input_ids = torch.tensor(input_ids)
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            if not hasattr(attention_mask, "detach"):
                attention_mask = torch.tensor(attention_mask)
            kwargs["attention_mask"] = attention_mask
        with torch.no_grad():
            out = self.model(**kwargs, use_cache=False)
        return out.logits.detach().cpu().float().numpy()

    def generate(self, prompt: str, **generation_kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with self.torch.no_grad():
            out_ids = self.model.generate(**inputs, **generation_kwargs)
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=False)
