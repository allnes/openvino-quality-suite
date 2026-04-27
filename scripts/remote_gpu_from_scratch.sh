#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-${HOME}/oviqs-gpu-from-scratch}"
REPO_DIR="${REPO_DIR:-${WORKDIR}/openvino-quality-suite}"
MODEL_ID="${MODEL_ID:-sshleifer/tiny-gpt2}"
MODEL_DIR="${MODEL_DIR:-${WORKDIR}/models/sshleifer--tiny-gpt2-eval_logits}"
REPORT_DIR="${REPORT_DIR:-${WORKDIR}/reports}"
DATASET="${DATASET:-${WORKDIR}/data/smoke_likelihood.jsonl}"
DEVICE="${DEVICE:-GPU}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-15}"
export PIP_RETRIES="${PIP_RETRIES:-0}"

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"
python3 -m venv .venv

.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -e "${REPO_DIR}[dev]"
.venv/bin/python -m pip install -r "${REPO_DIR}/requirements/gpu.txt"

mkdir -p "$(dirname "${DATASET}")" "${REPORT_DIR}" "$(dirname "${MODEL_DIR}")"
cat > "${DATASET}" <<'JSONL'
{"id":"smoke_001","task_type":"likelihood","text":"OpenVINO runs language model inference on Intel GPU."}
{"id":"smoke_002","task_type":"likelihood","text":"Quality metrics compare token likelihood and distribution drift."}
JSONL

.venv/bin/optimum-cli export openvino \
  --model "${MODEL_ID}" \
  --task text-generation \
  --weight-format fp16 \
  "${MODEL_DIR}"

.venv/bin/oviq eval-likelihood \
  --model "${MODEL_DIR}" \
  --backend openvino-runtime \
  --dataset "${DATASET}" \
  --device "${DEVICE}" \
  --out "${REPORT_DIR}/likelihood_gpu.json"

.venv/bin/oviq eval-drift \
  --reference "${MODEL_DIR}" \
  --current "${MODEL_DIR}" \
  --reference-backend openvino-runtime \
  --current-backend openvino-runtime \
  --reference-device "${DEVICE}" \
  --dataset "${DATASET}" \
  --device "${DEVICE}" \
  --out "${REPORT_DIR}/drift_gpu_self.json"

.venv/bin/oviq run-gpu-suite \
  --model "${MODEL_DIR}" \
  --backend openvino-runtime \
  --dataset "${DATASET}" \
  --device "${DEVICE}" \
  --out "${REPORT_DIR}/gpu_metric_suite.json"

.venv/bin/python -m pytest "${REPO_DIR}/tests/unit/test_openvino_runtime_runner.py"
.venv/bin/python -m ruff check "${REPO_DIR}"

echo "Reports written to ${REPORT_DIR}"
