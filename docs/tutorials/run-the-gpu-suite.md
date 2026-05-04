# Run the GPU suite

Use this tutorial when the model has already been exported to an OpenVINO
directory and you want a GPU-focused quality report. The GPU suite combines the
same report contract as CPU runs with device metadata and optional generation
checks.

## Prerequisites

Install the runtime extras required by the backend you want to exercise:

```bash
python -m pip install -e ".[openvino]"
python -m pip install -e ".[genai]"
```

Prepare a likelihood JSONL dataset. Keep it small for the first GPU smoke run:

```json
{"prompt":"The OpenVINO toolkit is","completion":" used for optimized inference."}
{"prompt":"Quality gates should keep","completion":" missing evidence visible."}
```

The `--model` path is used for OpenVINO Runtime checks that need logits. The
optional `--genai-model` path is used for OpenVINO GenAI generation checks. They
can point to the same exported model directory when the model supports both
surfaces in your environment.

## Run the suite

```bash
oviq run-gpu-suite \
  --model models/model-ov \
  --genai-model models/model-genai-ov \
  --dataset data/likelihood.jsonl \
  --out reports/gpu.json \
  --backend openvino-runtime \
  --device GPU
```

Use `GPU.0` instead of `GPU` when the machine exposes multiple GPU devices and
you need a deterministic target. Keep `--window-size` and `--stride` small for
smoke tests, then increase them for long-context checks.

## Validate and bundle

```bash
oviq report validate --report reports/gpu.json
oviq report build \
  --report reports/gpu.json \
  --out reports/gpu-bundle \
  --format all
```

The bundle should include `report.json`, `analysis.json`, `metrics.csv`,
`sample_metrics.jsonl`, `index.md`, `dashboard.html`, `metadata.json`, and
`assets/`.

## Interpret unknown metrics

An OpenVINO Runtime backend can support logits-level likelihood checks. A GenAI
generation path may be able to generate text but not expose aligned logits for a
specific metric. In that case the report should preserve the missing evidence as
`unknown` and include a finding that explains the unsupported surface.

Do not treat a GPU report as successful only because the command completed. The
review sequence is:

1. Confirm `run.device` records the intended GPU target.
2. Check `summary.overall_status`.
3. Inspect `analysis.json` for `unknown` and `fail` findings.
4. Use `metrics.csv` to compare scalar values against the CPU or baseline run.
5. Keep the generated bundle with the run logs used for reproduction.

## Common failures

If the command reports that no GPU device is available, validate the OpenVINO
runtime installation independently before changing gates. If the model loads on
CPU but fails on GPU, record the device, precision and model export parameters
in the report metadata. If only generation checks work, publish the report with
unknown logits metrics instead of replacing them with text-only approximations.
