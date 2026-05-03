from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class MetricPath:
    section: str
    name: str
    parts: tuple[str, ...] = ()

    @classmethod
    def parse(cls, value: str) -> MetricPath:
        if not value or "." not in value:
            raise ValueError(f"Metric path must be '<section>.<metric>', got {value!r}")
        section, *rest = value.split(".")
        if not section or not all(rest):
            raise ValueError(f"Metric path contains an empty segment: {value!r}")
        return cls(section=section, name=rest[-1], parts=tuple(rest))

    @classmethod
    def from_parts(cls, section: str, parts: tuple[str, ...]) -> MetricPath:
        if not section:
            raise ValueError("section must not be empty")
        if not parts or not all(parts):
            raise ValueError("metric path parts must not be empty")
        return cls(section=section, name=parts[-1], parts=parts)

    @property
    def dotted(self) -> str:
        return ".".join((self.section, *self.parts))


def section_title(section: str) -> str:
    return section.replace("_", " ").title()


def metric_path(section: str, *parts: str) -> str:
    return MetricPath.from_parts(section, tuple(parts)).dotted


__all__ = ["MetricPath", "metric_path", "section_title"]
