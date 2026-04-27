from __future__ import annotations

from oviqs.core.errors import OptionalDependencyError
from oviqs.core.runner import BaseGenerationRunner


class OVGenAIRunner(BaseGenerationRunner):
    name = "openvino-genai"

    def __init__(self, model_dir: str, device: str = "CPU") -> None:
        try:
            import openvino_genai as ov_genai
        except ImportError as exc:
            raise OptionalDependencyError("openvino-genai", "genai") from exc
        self.model_dir = model_dir
        self.device = device
        self.pipe = ov_genai.LLMPipeline(model_dir, device)

    def run_info(self):
        return {"backend": "openvino-genai", "model_dir": self.model_dir, "device": self.device}

    def generate(self, prompt: str, **generation_kwargs):
        return self.pipe.generate(prompt, **generation_kwargs)
