#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-${HOME}/oviqs-gpu-from-scratch}"
REPO_DIR="${REPO_DIR:-${WORKDIR}/openvino-quality-suite}"
REPORT_DIR="${REPORT_DIR:-${WORKDIR}/reports/target-models}"
DATASET="${DATASET:-${WORKDIR}/data/target_quality_micro.jsonl}"
DEVICE="${DEVICE:-GPU}"

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"
python3 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -e "${REPO_DIR}[dev]"
.venv/bin/python -m pip install -r "${REPO_DIR}/requirements/gpu.txt"
.venv/bin/python -m pip install --no-deps -r "${REPO_DIR}/requirements/gpu-target-models.txt"
.venv/bin/python -m pip install --no-deps -r "${REPO_DIR}/requirements/gpu-transformers-main.txt"

mkdir -p "$(dirname "${DATASET}")" "${REPORT_DIR}" "${WORKDIR}/models"
cat > "${DATASET}" <<'JSONL'
{"id":"qa_openvino_gpu","task_type":"likelihood","text":"OpenVINO compiles transformer language models for Intel GPU inference and reports quality metrics from logits."}
{"id":"rag_quality","task_type":"rag","text":"Retrieval augmented generation should cite the retrieved evidence and avoid unsupported claims."}
{"id":"agent_tool_use","task_type":"agent","text":"An assistant should call the search tool once, read the observation, and then answer with grounded facts."}
JSONL

run_model() {
  local name="$1"
  local model_id="$2"
  local weight_format="$3"
  local model_dir="${WORKDIR}/models/${name}-eval-${weight_format}"
  local genai_dir="${WORKDIR}/models/${name}-genai-${weight_format}"
  local report="${REPORT_DIR}/${name}_gpu_suite.json"

  {
    echo "== ${name}: export logits ${model_id} ${weight_format}"
    .venv/bin/optimum-cli export openvino \
      --model "${model_id}" \
      --task text-generation \
      --weight-format "${weight_format}" \
      "${model_dir}"

    echo "== ${name}: export genai ${model_id} ${weight_format}"
    if .venv/bin/optimum-cli export openvino \
      --model "${model_id}" \
      --task text-generation-with-past \
      --weight-format "${weight_format}" \
      "${genai_dir}"; then
      .venv/bin/oviq run-gpu-suite \
        --model "${model_dir}" \
        --backend openvino-runtime \
        --dataset "${DATASET}" \
        --device "${DEVICE}" \
        --genai-model "${genai_dir}" \
        --window-size 64 \
        --stride 32 \
        --out "${report}"
    else
      .venv/bin/oviq run-gpu-suite \
        --model "${model_dir}" \
        --backend openvino-runtime \
        --dataset "${DATASET}" \
        --device "${DEVICE}" \
        --window-size 64 \
        --stride 32 \
        --out "${report}"
    fi
  } >"${REPORT_DIR}/${name}.log" 2>&1 || {
    .venv/bin/python - <<PY
import json
from pathlib import Path

Path("${report}").write_text(json.dumps({
  "run": {
    "id": "${name}_gpu_suite",
    "model": "${model_id}",
    "device": "${DEVICE}",
    "current": "openvino-runtime",
    "suite": "openvino_llm_quality_v1_gpu"
  },
  "summary": {
    "overall_status": "fail",
    "main_findings": ["Export or GPU metric run failed. See ${REPORT_DIR}/${name}.log"]
  }
}, indent=2))
PY
  }
}

run_model gemma3_1b_it google/gemma-3-1b-it int4
run_model ministral3_3b_instruct_2512 mistralai/Ministral-3-3B-Instruct-2512 int4
run_model gpt_oss_20b openai/gpt-oss-20b int4
run_model qwen3_5_0_8b Qwen/Qwen3.5-0.8B int4

echo "Reports written to ${REPORT_DIR}"
