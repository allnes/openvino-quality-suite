# OpenVINO Runners

Use Optimum OpenVINO for a Hugging Face-like logits path and OpenVINO Runtime for lower
level plugin/device checks. Use OpenVINO GenAI for generation tests, not as the primary
source for full-distribution metrics.

## Backends

- `dummy`: deterministic local runner for tests and examples.
- `hf`: Hugging Face reference logits runner.
- `optimum-openvino`: Optimum Intel `OVModelForCausalLM` logits runner.
- `openvino-runtime`: lower-level OpenVINO Runtime logits runner.
- `openvino-genai`: generation-layer runner surface for output tests.
- `ovms-openai`: OpenAI-compatible serving surface for OVMS checks.

## Export pattern

Keep separate exported artifacts for evaluation and production when needed:

- Evaluation export: no KV-cache requirement, optimized for full-sequence logits.
- Production export: KV-cache enabled, quantization and serving configuration match deployment.

Full-distribution diagnostics should compare reference and current logits over the same
token IDs and masks. Generation-only APIs are useful for smoke and application-level
checks, but they are not a replacement for logits-level equivalence.

## GPU Runtime Notes

`OVRuntimeLogitsRunner` accepts either an OpenVINO `.xml` path or an exported model
directory. For directory inputs it finds `openvino_model.xml` and loads the tokenizer from
the same directory.

On GPU it reshapes the model to the actual `input_ids`, `attention_mask` and
`position_ids` shapes before compiling. This avoids compiling fully dynamic LLM shapes
when running metric-sized samples.
