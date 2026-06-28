#!/usr/bin/env python3
"""End-to-end GPU smoke: prepare a model, then run likelihood, drift and the suite.

Two model-preparation modes (env var ``MODE`` or ``--mode``):

* ``openvino`` (default): download the ready-made OpenVINO INT4 artifact.
* ``convert``: direct conversion of the base checkpoint via ``optimum-cli``.

Paths/device/ids overridable via environment variables (WORKDIR, REPO_DIR,
MODE, BASE_ID, ARTIFACT, WEIGHT_FORMAT, MODEL_DIR, REPORT_DIR, DATASET, DEVICE).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DATASET_ROWS = [
    {
        "id": "smoke_001",
        "task_type": "likelihood",
        "text": "OpenVINO runs language model inference on Intel GPU.",
    },
    {
        "id": "smoke_002",
        "task_type": "likelihood",
        "text": "Quality metrics compare token likelihood and distribution drift.",
    },
]


def run(cmd: list[str], *, env: dict | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", default=os.environ.get("MODE", "openvino"), choices=["openvino", "convert"]
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", "GPU"))
    args = parser.parse_args()

    os.environ.setdefault("PIP_DEFAULT_TIMEOUT", "15")
    os.environ.setdefault("PIP_RETRIES", "0")

    home = Path.home()
    workdir = Path(os.environ.get("WORKDIR", home / "oviqs-gpu-from-scratch"))
    repo_dir = Path(os.environ.get("REPO_DIR", workdir / "openvino-quality-suite"))
    base_id = os.environ.get("BASE_ID", "Qwen/Qwen3-0.6B")
    artifact = os.environ.get("ARTIFACT", "OpenVINO/Qwen3-0.6B-int4-ov")
    weight_format = os.environ.get("WEIGHT_FORMAT", "int4")
    model_dir = Path(os.environ.get("MODEL_DIR", workdir / "models" / "qwen3-0_6b-int4-ov"))
    report_dir = Path(os.environ.get("REPORT_DIR", workdir / "reports"))
    dataset = Path(os.environ.get("DATASET", workdir / "data" / "smoke_likelihood.jsonl"))

    def venv_bin(tool: str) -> str:
        return str(workdir / ".venv" / "bin" / tool)

    workdir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "venv", str(workdir / ".venv")])
    py = venv_bin("python")
    run([py, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
    run([py, "-m", "pip", "install", "-e", f"{repo_dir}[dev]"])
    run([py, "-m", "pip", "install", "-r", str(repo_dir / "requirements" / "gpu.txt")])

    dataset.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text("\n".join(json.dumps(row) for row in DATASET_ROWS) + "\n", encoding="utf-8")

    if args.mode == "convert":
        run(
            [
                venv_bin("optimum-cli"),
                "export",
                "openvino",
                "--model",
                base_id,
                "--task",
                "text-generation",
                "--weight-format",
                weight_format,
                str(model_dir),
            ]
        )
    else:
        run([venv_bin("huggingface-cli"), "download", artifact, "--local-dir", str(model_dir)])

    run(
        [
            venv_bin("oviq"),
            "eval-likelihood",
            "--model",
            str(model_dir),
            "--backend",
            "openvino-runtime",
            "--dataset",
            str(dataset),
            "--device",
            args.device,
            "--out",
            str(report_dir / "likelihood_gpu.json"),
        ]
    )

    run(
        [
            venv_bin("oviq"),
            "eval-drift",
            "--reference",
            str(model_dir),
            "--current",
            str(model_dir),
            "--reference-backend",
            "openvino-runtime",
            "--current-backend",
            "openvino-runtime",
            "--reference-device",
            args.device,
            "--dataset",
            str(dataset),
            "--device",
            args.device,
            "--out",
            str(report_dir / "drift_gpu_self.json"),
        ]
    )

    run(
        [
            venv_bin("oviq"),
            "run-gpu-suite",
            "--model",
            str(model_dir),
            "--backend",
            "openvino-runtime",
            "--dataset",
            str(dataset),
            "--device",
            args.device,
            "--out",
            str(report_dir / "gpu_metric_suite.json"),
        ]
    )

    run([py, "-m", "pytest", str(repo_dir / "tests" / "unit" / "test_openvino_runtime_runner.py")])
    run([py, "-m", "ruff", "check", str(repo_dir)])

    print(f"Reports written to {report_dir} (MODE={args.mode})")


if __name__ == "__main__":
    main()
