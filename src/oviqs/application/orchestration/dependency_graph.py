from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DependencyGraph:
    edges: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def dependencies_for(self, node: str) -> tuple[str, ...]:
        return self.edges.get(node, ())
