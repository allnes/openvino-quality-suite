#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-${HOME}/oviqs-gpu-from-scratch}"
REPO_DIR="${REPO_DIR:-${WORKDIR}/openvino-quality-suite}"
MODEL_ID="${MODEL_ID:-openai-community/gpt2}"
MODEL_DIR="${MODEL_DIR:-${WORKDIR}/models/openai-community--gpt2-eval-fp16}"
GENAI_MODEL_DIR="${GENAI_MODEL_DIR:-${WORKDIR}/models/openai-community--gpt2-genai-fp16}"
REPORT="${REPORT:-${WORKDIR}/reports/target-models/gpt2_gpu_suite.json}"
DATASET="${DATASET:-${WORKDIR}/data/target_quality_micro.jsonl}"
DEVICE="${DEVICE:-GPU}"

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"
python3 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -r "${REPO_DIR}/requirements/gpu.txt"

mkdir -p "$(dirname "${DATASET}")" "$(dirname "${REPORT}")" "$(dirname "${MODEL_DIR}")" logs
cat > "${DATASET}" <<'JSONL'
{"id":"qa_openvino_gpu","task_type":"likelihood","text":"OpenVINO compiles transformer language models for Intel GPU inference and reports quality metrics from logits."}
{"id":"rag_quality","task_type":"rag","text":"Retrieval augmented generation should cite the retrieved evidence and avoid unsupported claims."}
{"id":"agent_tool_use","task_type":"agent","text":"An assistant should call the search tool once, read the observation, and then answer with grounded facts."}
JSONL

.venv/bin/optimum-cli export openvino \
  --model "${MODEL_ID}" \
  --task text-generation \
  --weight-format fp16 \
  "${MODEL_DIR}" 2>&1 | tee logs/gpt2_eval_export.log

.venv/bin/optimum-cli export openvino \
  --model "${MODEL_ID}" \
  --task text-generation-with-past \
  --weight-format fp16 \
  "${GENAI_MODEL_DIR}" 2>&1 | tee logs/gpt2_genai_export.log

PYTHONPATH="${REPO_DIR}/src" .venv/bin/oviq run-gpu-suite \
  --model "${MODEL_DIR}" \
  --backend openvino-runtime \
  --dataset "${DATASET}" \
  --device "${DEVICE}" \
  --genai-model "${GENAI_MODEL_DIR}" \
  --window-size 64 \
  --stride 32 \
  --out "${REPORT}"

echo "Validated GPU report written to ${REPORT}"
