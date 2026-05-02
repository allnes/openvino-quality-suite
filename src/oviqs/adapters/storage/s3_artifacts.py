from __future__ import annotations


class S3ArtifactStorage:
    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError("S3 artifact storage is not implemented in v0.1.0.")


__all__ = ["S3ArtifactStorage"]
