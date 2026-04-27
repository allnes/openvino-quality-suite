class OVIQSError(Exception):
    """Base exception for OVIQS."""


class OptionalDependencyError(ImportError):
    """Raised when an optional backend dependency is missing."""

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f"Missing optional dependency '{package}'. Install with: pip install 'oviqs[{extra}]'"
        )
