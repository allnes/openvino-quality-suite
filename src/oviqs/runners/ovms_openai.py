from __future__ import annotations

import json
import time
from urllib import request
from urllib.parse import urlparse

from oviqs.core.runner import BaseGenerationRunner
from oviqs.core.types import GenerationOutput


class OVMSOpenAIRunner(BaseGenerationRunner):
    name = "ovms-openai"

    def __init__(self, base_url: str, model: str, timeout_s: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("OVMS base_url must use http or https")
        self.model = model
        self.timeout_s = timeout_s

    def run_info(self) -> dict:
        return {"backend": "ovms-openai", "base_url": self.base_url, "model": self.model}

    def generate(self, prompt: str, **generation_kwargs) -> GenerationOutput:
        payload = {"model": self.model, "prompt": prompt, **generation_kwargs}
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/v3/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.perf_counter()
        with request.urlopen(req, timeout=self.timeout_s) as resp:  # nosec B310
            body = json.loads(resp.read().decode("utf-8"))
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = body.get("choices", [{}])[0].get("text", "")
        return GenerationOutput(text=text, latency_ms=latency_ms, metadata={"response": body})
