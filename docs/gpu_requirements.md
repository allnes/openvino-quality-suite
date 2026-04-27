# GPU Requirements

This page describes the public GPU validation setup for OVIQS. It intentionally avoids
machine-specific paths, private workspace names and generated report payloads. Keep local
reports under ignored directories such as `reports/`, `models/`, `data/` or a workspace
outside the repository.

GPU validation dependencies are tracked in:

- `requirements/gpu.txt` for OpenVINO Runtime, OpenVINO GenAI and export tooling.
- `requirements/gpu-target-models.txt` for target model families that need newer package
  support.
- `requirements/gpu-transformers-main.txt` for experimental validation against bleeding
  edge Transformers support.

## Clean Workspace

Use a disposable workspace outside the repository checkout:

```bash
export WORKDIR="${WORKDIR:-/tmp/oviqs-gpu-workspace}"
export REPO_DIR="${REPO_DIR:-${WORKDIR}/openvino-quality-suite}"
mkdir -p "${WORKDIR}"
```

Clone or synchronize the repository to `${REPO_DIR}`, then create a virtual environment:

```bash
python3 -m venv "${WORKDIR}/.venv"
"${WORKDIR}/.venv/bin/python" -m pip install -U pip setuptools wheel
"${WORKDIR}/.venv/bin/python" -m pip install -e "${REPO_DIR}[dev]"
"${WORKDIR}/.venv/bin/python" -m pip install -r "${REPO_DIR}/requirements/gpu.txt"
```

Install extended requirements only when validating model families that need them:

```bash
"${WORKDIR}/.venv/bin/python" -m pip install --no-deps \
  -r "${REPO_DIR}/requirements/gpu-target-models.txt"
"${WORKDIR}/.venv/bin/python" -m pip install --no-deps \
  -r "${REPO_DIR}/requirements/gpu-transformers-main.txt"
```

The requirements intentionally use CPU PyTorch wheels for export/orchestration. Converted
OpenVINO artifacts are executed by OpenVINO Runtime or OpenVINO GenAI on the selected Intel
GPU device.

## Validation Architecture

Use separate artifacts when possible:

- `text-generation` export for logits-level metrics through OpenVINO Runtime.
- `text-generation-with-past` export for generation and KV-cache checks through OpenVINO
  GenAI or a stateful OpenVINO Runtime runner.

Use separate metric sections:

- `likelihood`: NLL, PPL and token counts.
- `inference_equivalence`: KL, JS, entropy drift, logit cosine, top-k overlap and top-1
  change rate.
- `long_context`: sliding-window PPL, position buckets, lost-in-the-middle score and
  robustness checks.
- `serving`: batch invariance and KV-cache drift when a stateful IR is supplied.
- `generation`: deterministic output checks through a generation-capable runner.
- `rag` and `agent`: deterministic local checks plus explicit `unknown` values for
  judge-based metrics when no external scorer is configured.

## Required Metadata

Every GPU validation report should capture enough metadata to reproduce the run:

- repository commit;
- model ID and local artifact name;
- export backend and weight format;
- OpenVINO Runtime/OpenVINO GenAI versions;
- device name, device ID and driver/runtime versions;
- dataset name, dataset revision or local fixture hash;
- prompt length/output length buckets;
- gates file and suite config.

Do not commit generated report JSON, model artifacts, dataset caches, logs or private
machine inventory. Put them under ignored directories or publish sanitized aggregate
results separately.

## Minimal Smoke Run

The repository includes a clean smoke script:

```bash
WORKDIR=/tmp/oviqs-gpu-workspace \
REPO_DIR=/path/to/openvino-quality-suite \
bash scripts/remote_gpu_from_scratch.sh
```

The script creates a local dataset in `${WORKDIR}/data`, exports a tiny model into
`${WORKDIR}/models`, runs GPU metrics and writes reports under `${WORKDIR}/reports`.

## Target Model Run

For broader target model coverage:

```bash
WORKDIR=/tmp/oviqs-gpu-workspace \
REPO_DIR=/path/to/openvino-quality-suite \
bash scripts/remote_gpu_target_models.sh
```

Use `configs/examples/genai_metric_models.yaml` to inspect recommended model tiers and
export variants:

```bash
oviq list-genai-models --tier smoke --metric likelihood
oviq genai-export-plan --model Qwen/Qwen2.5-0.5B-Instruct
```

## Publication Rules

Before publishing GPU validation work:

- keep raw reports, logs, datasets and model artifacts out of Git;
- remove local usernames, hostnames, mount points and cloud-drive paths;
- replace private paths with environment variables such as `${WORKDIR}` and `${REPO_DIR}`;
- summarize results in docs only when they are reproducible, sanitized and tied to a
  public commit/config;
- leave unsupported or unavailable judge-based metrics as `unknown` instead of fabricating
  pass/fail values.
