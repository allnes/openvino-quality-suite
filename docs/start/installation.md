# Installation

Use the project virtual environment for development and documentation work:

```bash
uv sync --extra dev
```

That installs the package, the documentation toolchain, type checkers, security
tools and test dependencies used by the repository quality gate.

For a local editable install with only the default runtime dependencies:

```bash
python -m pip install -e .
```

Optional OpenVINO, GenAI, evaluation, RAG, agent, observability and plot
dependencies are declared as package extras. Install only the extras required for
the backend you are validating:

```bash
python -m pip install -e ".[openvino]"
python -m pip install -e ".[genai]"
python -m pip install -e ".[rag,agent]"
```

Use `uv run` or binaries from `.venv/bin` when running checks from the repo:

```bash
uv run pytest -q
uv run mkdocs build --strict
uv run pre-commit run --all-files --show-diff-on-failure
```

If an optional backend is unavailable, keep the missing metrics as `unknown` in
reports. Do not replace unsupported evidence with approximations just to make a
gate pass.
