from __future__ import annotations


class WhitespaceTokenizer:
    """Tiny deterministic tokenizer for tests and dummy examples."""

    def encode(self, text: str) -> list[int]:
        return [abs(hash(token)) % 32000 for token in text.split()]
