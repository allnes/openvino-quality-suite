from __future__ import annotations

import numpy as np

from oviqs.core.runner import BaseGenerationRunner, BaseLogitsRunner


class DummyLogitsRunner(BaseLogitsRunner):
    name = "dummy-logits"

    def __init__(self, vocab_size: int = 16, correct_bias: float = 5.0) -> None:
        self.vocab_size = vocab_size
        self.correct_bias = correct_bias

    def run_info(self) -> dict:
        return {"backend": "dummy", "vocab_size": self.vocab_size}

    def forward_logits(self, input_ids, attention_mask=None):
        ids = np.asarray(input_ids, dtype=np.int64)
        logits = np.zeros((*ids.shape, self.vocab_size), dtype=np.float32)
        if ids.shape[1] > 1:
            next_ids = ids[:, 1:]
            for batch_idx in range(ids.shape[0]):
                for pos in range(ids.shape[1] - 1):
                    logits[batch_idx, pos, next_ids[batch_idx, pos] % self.vocab_size] = (
                        self.correct_bias
                    )
        return logits


class DummyGenerationRunner(BaseGenerationRunner):
    name = "dummy-generation"

    def run_info(self) -> dict:
        return {"backend": "dummy"}

    def generate(self, prompt: str, **generation_kwargs):
        return f"{prompt} [dummy-completion]"
