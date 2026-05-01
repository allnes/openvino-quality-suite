from __future__ import annotations

from typing import Any, Protocol


class TokenizerPort(Protocol):
    def encode(self, text: str) -> dict[str, Any]: ...

    def decode(self, _token_ids: list[int]) -> str: ...
