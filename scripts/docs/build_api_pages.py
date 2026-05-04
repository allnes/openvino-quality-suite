from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "oviqs"
OUT_DIR = ROOT / "docs" / "reference" / "api"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    modules = []
    expected_pages: set[Path] = set()
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        module = ".".join(path.relative_to(ROOT / "src").with_suffix("").parts)
        target = OUT_DIR / f"{module.replace('.', '/')}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# `{module}`\n\n::: {module}\n", encoding="utf-8")
        expected_pages.add(target)
        modules.append((module, target.relative_to(OUT_DIR)))
    expected_pages.add(OUT_DIR / "index.md")
    for stale_page in OUT_DIR.rglob("*.md"):
        if stale_page not in expected_pages:
            stale_page.unlink()
    lines = [
        "# API reference",
        "",
        "Generated mkdocstrings stubs for the current checkout.",
        "",
        "OVIQS is pre-stable. Treat the JSON schemas, report bundle layout and CLI as "
        "the public contracts; Python modules are documented for extension work and may "
        "move before the first stable release.",
        "",
        "Use the package layers when choosing an import target:",
        "",
        "- `oviqs.domain` contains backend-independent models, metrics and rules.",
        "- `oviqs.application` orchestrates use cases and owns service-level DTOs.",
        "- `oviqs.ports` defines interfaces implemented by adapters.",
        "- `oviqs.adapters` integrates filesystems, runners, renderers and optional tools.",
        "- `oviqs.interfaces` exposes CLI, HTTP and gRPC entry points.",
        "",
        "Prefer domain and application imports for extension tests. Adapter imports should "
        "stay at the boundary where optional dependencies are available.",
        "",
    ]
    lines.extend(f"- [`{module}`]({path.as_posix()})" for module, path in modules)
    (OUT_DIR / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
