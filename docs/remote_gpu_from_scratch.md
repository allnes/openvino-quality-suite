# Remote GPU From Scratch

Use this flow inside a clean workspace on the target GPU machine. Do not reuse
pre-existing model artifacts, private datasets or unrelated folders.

Default workspace:

```bash
export WORKDIR="${WORKDIR:-/tmp/oviqs-gpu-workspace}"
export REPO_DIR="${REPO_DIR:-${WORKDIR}/openvino-quality-suite}"
```

Run after the repository has been cloned or synchronized to `${REPO_DIR}`:

```bash
WORKDIR="${WORKDIR}" REPO_DIR="${REPO_DIR}" bash "${REPO_DIR}/scripts/remote_gpu_from_scratch.sh"
```

If the target machine needs a proxy, provide it through standard environment variables
when invoking the script. Do not commit proxy hostnames or credentials to the repository.

The script:

1. Creates `${WORKDIR}/.venv`.
2. Installs OVIQS and `requirements/gpu.txt` into that venv.
3. Writes a tiny JSONL likelihood dataset under `${WORKDIR}/data`.
4. Converts `sshleifer/tiny-gpt2` to OpenVINO IR under `${WORKDIR}/models`.
5. Runs `oviq eval-likelihood` on `GPU`.
6. Runs self-drift with OpenVINO Runtime on `GPU`.
7. Runs `oviq run-gpu-suite` to produce a multi-section GPU scorecard.
8. Runs focused unit/lint checks.

Generated models, datasets, logs and reports must stay outside the repository or under
ignored paths. The default script writes them under `${WORKDIR}`.

## Requirements

Install requirements only inside the target GPU-machine virtual environment:

```bash
python3 -m venv "${WORKDIR}/.venv"
"${WORKDIR}/.venv/bin/python" -m pip install -U pip setuptools wheel
"${WORKDIR}/.venv/bin/python" -m pip install -e "${REPO_DIR}[dev]"
"${WORKDIR}/.venv/bin/python" -m pip install -r "${REPO_DIR}/requirements/gpu.txt"
```

The file intentionally uses CPU PyTorch wheels for export/orchestration. OpenVINO Runtime
executes converted IR on the Intel GPU.

## GPU Suite Scope

`oviq run-gpu-suite` follows the project architecture:

- logits-level layer on OpenVINO Runtime GPU: NLL, PPL, self KL/JS, entropy drift,
  logit cosine, sliding-window PPL, position buckets and batch invariance drift;
- generation layer through OpenVINO GenAI only when a `--genai-model` export is provided;
- RAG and agent sections report deterministic local checks and mark judge-based metrics as
  `unknown` when no external scorer is configured.
