from __future__ import annotations


class OptionalDependencyError(ImportError):
    """Raised when an optional adapter dependency is missing."""

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f"Missing optional dependency '{package}'. Install with: pip install 'oviqs[{extra}]'"
        )


__all__ = ["OptionalDependencyError"]
