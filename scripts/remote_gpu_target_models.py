#!/usr/bin/env python3
"""Run the OVIQS GPU metric suite over the five canonical networks.

Two model-preparation modes (env var ``MODE`` or ``--mode``):

* ``openvino`` (default): download the ready-made OpenVINO INT4 IR artifact from
  the OpenVINO/llm collection. No export step.
* ``convert``: direct conversion of the base Hugging Face checkpoint to OpenVINO
  IR with ``optimum-cli`` (``text-generation`` for logits,
  ``text-generation-with-past`` for the GenAI generation layer).

All paths/device/mode are overridable via environment variables to match the
previous shell entry point (WORKDIR, REPO_DIR, REPORT_DIR, DATASET, DEVICE, MODE).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# name, base HF checkpoint, OpenVINO artifact, weight format
TARGET_MODELS = [
    ("qwen3_0_6b", "Qwen/Qwen3-0.6B", "OpenVINO/Qwen3-0.6B-int4-ov", "int4"),
    ("gemma2_9b_it", "google/gemma-2-9b-it", "OpenVINO/gemma-2-9b-it-int4-ov", "int4"),
    (
        "mistral7b",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "OpenVINO/mistral-7b-instruct-v0.1-int4-ov",
        "int4",
    ),
    ("gpt_oss_20b", "openai/gpt-oss-20b", "OpenVINO/gpt-oss-20b-int4-ov", "int4"),
    ("phi_4", "microsoft/phi-4", "OpenVINO/phi-4-int4-ov", "int4"),
]

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
    """Run a command, streaming to a log file when given (raises on failure)."""
    if log is not None:
        with log.open("ab") as handle:
            handle.write(f"\n$ {' '.join(cmd)}\n".encode())
            handle.flush()
            subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True)


def venv_bin(workdir: Path, tool: str) -> str:
    return str(workdir / ".venv" / "bin" / tool)


def prepare_model(
    *,
    workdir: Path,
    mode: str,
    name: str,
    base_id: str,
    artifact: str,
    weight_format: str,
    log: Path,
) -> tuple[Path, Path]:
    """Return (model_dir, genai_dir) for the run, preparing the artifacts."""
    if mode == "convert":
        model_dir = workdir / "models" / f"{name}-eval-{weight_format}"
        genai_dir = workdir / "models" / f"{name}-genai-{weight_format}"
        run(
            [
                venv_bin(workdir, "optimum-cli"),
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
            log=log,
        )
        run(
            [
                venv_bin(workdir, "optimum-cli"),
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
            log=log,
        )
        return model_dir, genai_dir
    model_dir = workdir / "models" / f"{name}-int4-ov"
    run(
        [venv_bin(workdir, "huggingface-cli"), "download", artifact, "--local-dir", str(model_dir)],
        log=log,
    )
    return model_dir, model_dir


def run_model(
    *,
    workdir: Path,
    mode: str,
    device: str,
    report_dir: Path,
    dataset: Path,
    name: str,
    base_id: str,
    artifact: str,
    weight_format: str,
) -> None:
    report = report_dir / f"{name}_gpu_suite.json"
    log = report_dir / f"{name}.log"
    try:
        print(f"== {name}: prepare model (MODE={mode})")
        model_dir, genai_dir = prepare_model(
            workdir=workdir,
            mode=mode,
            name=name,
            base_id=base_id,
            artifact=artifact,
            weight_format=weight_format,
            log=log,
        )
        print(f"== {name}: run gpu suite (model={model_dir} genai={genai_dir})")
        run(
            [
                venv_bin(workdir, "oviq"),
                "run-gpu-suite",
                "--model",
                str(model_dir),
                "--backend",
                "openvino-runtime",
                "--dataset",
                str(dataset),
                "--device",
                device,
                "--genai-model",
                str(genai_dir),
                "--window-size",
                "64",
                "--stride",
                "32",
                "--out",
                str(report),
            ],
            log=log,
        )
    except subprocess.CalledProcessError:
        report.write_text(
            json.dumps(
                {
                    "run": {
                        "id": f"{name}_gpu_suite",
                        "model": base_id,
                        "openvino_artifact": artifact,
                        "mode": mode,
                        "device": device,
                        "current": "openvino-runtime",
                        "suite": "openvino_llm_quality_v1_gpu",
                    },
                    "summary": {
                        "overall_status": "fail",
                        "main_findings": [
                            f"Model prepare (MODE={mode}) or GPU metric run failed. See {log}"
                        ],
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )


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
    report_dir = Path(os.environ.get("REPORT_DIR", workdir / "reports" / "target-models"))
    dataset = Path(os.environ.get("DATASET", workdir / "data" / "target_quality_micro.jsonl"))

    workdir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "venv", str(workdir / ".venv")])
    pip = venv_bin(workdir, "python")
    run([pip, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
    run([pip, "-m", "pip", "install", "-e", f"{repo_dir}[dev]"])
    run([pip, "-m", "pip", "install", "-r", str(repo_dir / "requirements" / "gpu.txt")])
    run(
        [
            pip,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "-r",
            str(repo_dir / "requirements" / "gpu-target-models.txt"),
        ]
    )
    run(
        [
            pip,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "-r",
            str(repo_dir / "requirements" / "gpu-transformers-main.txt"),
        ]
    )

    dataset.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (workdir / "models").mkdir(parents=True, exist_ok=True)
    dataset.write_text("\n".join(json.dumps(row) for row in DATASET_ROWS) + "\n", encoding="utf-8")

    for name, base_id, artifact, weight_format in TARGET_MODELS:
        run_model(
            workdir=workdir,
            mode=args.mode,
            device=args.device,
            report_dir=report_dir,
            dataset=dataset,
            name=name,
            base_id=base_id,
            artifact=artifact,
            weight_format=weight_format,
        )

    print(f"Reports written to {report_dir} (MODE={args.mode})")


if __name__ == "__main__":
    main()
