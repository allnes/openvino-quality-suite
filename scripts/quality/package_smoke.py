from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = REPO_ROOT / "dist"


def run(command: list[str], *, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def find_wheel() -> Path:
    wheels = sorted(DIST_DIR.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"No wheel was produced in {DIST_DIR}")
    if len(wheels) != 1:
        print(f"Found multiple wheels, using the last one: {wheels[-1]}")
    return wheels[-1]


def main() -> int:
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)

    run([sys.executable, "-m", "build"])
    run([sys.executable, "-m", "twine", "check", *map(str, DIST_DIR.glob("*"))])

    wheel = find_wheel()

    with tempfile.TemporaryDirectory(prefix="oviqs-wheel-smoke-") as tmp:
        venv_dir = Path(tmp) / "venv"
        run([sys.executable, "-m", "venv", str(venv_dir)])

        if os.name == "nt":
            python_bin = venv_dir / "Scripts" / "python.exe"
            oviq_bin = venv_dir / "Scripts" / "oviq.exe"
        else:
            python_bin = venv_dir / "bin" / "python"
            oviq_bin = venv_dir / "bin" / "oviq"

        run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(python_bin), "-m", "pip", "install", "--force-reinstall", str(wheel)])
        run([str(python_bin), "-c", "import oviqs"])
        run([str(oviq_bin), "--help"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
