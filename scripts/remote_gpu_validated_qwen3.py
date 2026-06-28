#!/usr/bin/env python3
"""Validated single-model GPU sanity run on the smallest canonical network.

Qwen3-0.6B, in one of two model-preparation modes (env var ``MODE`` or ``--mode``):

* ``openvino`` (default): download the ready-made OpenVINO INT4 IR artifact.
* ``convert``: direct conversion of the base checkpoint via ``optimum-cli``
  (``text-generation`` + ``text-generation-with-past``).

Paths/device/ids are overridable via environment variables (WORKDIR, REPO_DIR,
BASE_ID, ARTIFACT, WEIGHT_FORMAT, REPORT, DATASET, DEVICE, MODE).
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
        "id": "qa_openvino_gpu",
        "task_type": "likelihood",
        "text": (
            "OpenVINO compiles transformer language models for Intel GPU inference "
            "and reports quality metrics from logits."
        ),
    },
    {
        "id": "rag_quality",
        "task_type": "rag",
        "text": (
            "Retrieval augmented generation should cite the retrieved evidence "
            "and avoid unsupported claims."
        ),
    },
    {
        "id": "agent_tool_use",
        "task_type": "agent",
        "text": (
            "An assistant should call the search tool once, read the observation, "
            "and then answer with grounded facts."
        ),
    },
]


def run(cmd: list[str], *, log: Path | None = None) -> None:
    if log is not None:
        with log.open("ab") as handle:
            handle.write(f"\n$ {' '.join(cmd)}\n".encode())
            handle.flush()
            subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", default=os.environ.get("MODE", "openvino"), choices=["openvino", "convert"]
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", "GPU"))
    args = parser.parse_args()

    home = Path.home()
    workdir = Path(os.environ.get("WORKDIR", home / "oviqs-gpu-from-scratch"))
    repo_dir = Path(os.environ.get("REPO_DIR", workdir / "openvino-quality-suite"))
    base_id = os.environ.get("BASE_ID", "Qwen/Qwen3-0.6B")
    artifact = os.environ.get("ARTIFACT", "OpenVINO/Qwen3-0.6B-int4-ov")
    weight_format = os.environ.get("WEIGHT_FORMAT", "int4")
    report = Path(
        os.environ.get(
            "REPORT", workdir / "reports" / "target-models" / "qwen3_0_6b_gpu_suite.json"
        )
    )
    dataset = Path(os.environ.get("DATASET", workdir / "data" / "target_quality_micro.jsonl"))

    def venv_bin(tool: str) -> str:
        return str(workdir / ".venv" / "bin" / tool)

    workdir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "venv", str(workdir / ".venv")])
    py = venv_bin("python")
    run([py, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
    run([py, "-m", "pip", "install", "-r", str(repo_dir / "requirements" / "gpu.txt")])

    dataset.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    (workdir / "models").mkdir(parents=True, exist_ok=True)
    logs = workdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    dataset.write_text("\n".join(json.dumps(row) for row in DATASET_ROWS) + "\n", encoding="utf-8")

    if args.mode == "convert":
        model_dir = Path(
            os.environ.get("MODEL_DIR", workdir / "models" / f"qwen3-0_6b-eval-{weight_format}")
        )
        genai_dir = Path(
            os.environ.get(
                "GENAI_MODEL_DIR", workdir / "models" / f"qwen3-0_6b-genai-{weight_format}"
            )
        )
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
            ],
            log=logs / "qwen3_eval_export.log",
        )
        run(
            [
                venv_bin("optimum-cli"),
                "export",
                "openvino",
                "--model",
                base_id,
                "--task",
                "text-generation-with-past",
                "--weight-format",
                weight_format,
                str(genai_dir),
            ],
            log=logs / "qwen3_genai_export.log",
        )
    else:
        model_dir = Path(os.environ.get("MODEL_DIR", workdir / "models" / "qwen3-0_6b-int4-ov"))
        genai_dir = model_dir
        run(
            [venv_bin("huggingface-cli"), "download", artifact, "--local-dir", str(model_dir)],
            log=logs / "qwen3_download.log",
        )

    env = dict(os.environ, PYTHONPATH=str(repo_dir / "src"))
    subprocess.run(
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
            "--genai-model",
            str(genai_dir),
            "--window-size",
            "64",
            "--stride",
            "32",
            "--out",
            str(report),
        ],
        check=True,
        env=env,
    )

    print(f"Validated GPU report written to {report} (MODE={args.mode})")


if __name__ == "__main__":
    main()
