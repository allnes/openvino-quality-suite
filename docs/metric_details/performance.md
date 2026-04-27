# Performance Metrics

Performance metrics measure runtime cost under fixed model, device, dataset and decoding
conditions. They are not quality metrics, but every quality comparison should record them
because optimizations often trade latency, throughput, memory and numerical stability.

Use them to answer:

- Did a model export, precision or OpenVINO setting change latency?
- Did throughput improve without breaking quality gates?
- Are tail latencies stable enough for serving?
- Is generation latency explained by token count, KV-cache behavior or device settings?

## Scientific Basis

Inference benchmarks usually report both latency and throughput because optimizing one can
hurt the other. Batch size, concurrency, prompt length, output length, device, precision,
warmup, caching and scheduling all affect the result. A performance number without those
conditions is not reproducible.

OVIQS uses simple deterministic summaries:

```text
mean_latency = mean(latency_ms)
p95_latency = percentile_95(latency_ms)
tokens_per_second = num_tokens / (latency_ms / 1000)
```

## OVIQS Implementation

Source: `src/oviqs/metrics/performance.py` and runner metadata.

Current helper:

- `tokens_per_second(num_tokens, latency_ms)`.

Other performance fields are emitted by runners and reports:

- `forward_latency_ms_mean`;
- `forward_latency_ms_p95`;
- `tokens_per_second_forward`;
- `generation_latency_ms`.

## Metrics

### forward_latency_ms_mean

Formula:

```text
forward_latency_ms_mean = mean(forward_pass_latency_ms)
```

Direction: lower is better.

Interpretation:

- Mean latency captures average forward-pass cost.
- It can hide slow outliers, so always review p95 with it.
- Compare only after warmup and with the same input shape/bucket.

Action:

- Check dynamic shape compilation and whether first-run compile time leaked into samples.
- Compare precision and device settings.
- Bucket by prompt length and batch size.

### forward_latency_ms_p95

Formula:

```text
forward_latency_ms_p95 = percentile_95(forward_pass_latency_ms)
```

Direction: lower is better.

Interpretation:

- Captures tail latency.
- Important for user-facing serving because p95 can regress while mean stays stable.
- Sensitive to system noise, concurrent load and thermal/device state.

Action:

- Run enough samples per bucket.
- Separate cold start, warmup and steady-state measurements.
- Inspect outlier sample IDs and lengths.

### tokens_per_second_forward

Formula:

```text
tokens_per_second_forward =
  num_tokens / (forward_latency_ms / 1000)
```

Direction: higher is better.

Interpretation:

- Throughput normalizes latency by token count.
- It is not comparable across different prompt lengths, batch sizes or hardware.
- A throughput improvement with quality drift may be unacceptable; check gates together.

Action:

- Report batch size and token count.
- Compare short, medium and long prompt buckets separately.
- Use the same warmup and measurement window.

### generation_latency_ms

Formula:

```text
generation_latency_ms = wall_clock_end_to_end_generation_duration_ms
```

Direction: lower is better for a fixed output length and decoding configuration.

Interpretation:

- Includes prefill, decode, tokenization/detokenization and runtime overhead depending on
  the runner.
- Must be interpreted with generated token count and decoding parameters.
- If generation latency regresses while forward latency is stable, inspect KV-cache,
  tokenizer, stop conditions and serving overhead.

Related GenAI serving metrics that may be present in runner metadata:

- TTFT: time to first token;
- TPOT: time per output token;
- IPOT: inference time per output token;
- throughput in tokens per second.

## Dataset Design

Minimal performance row:

```json
{"id":"p1","task_type":"performance","prompt":"Summarize the release note.","max_new_tokens":64}
```

Recommended datasets:

- fixed prompt set with stable token counts;
- separate short, medium and long prompt buckets;
- fixed output lengths for generation latency;
- representative production prompts for final release checks.

Dataset rules:

- Record model artifact, device, driver/runtime, precision, OpenVINO version and backend.
- Record prompt tokens, generated tokens, batch size, concurrency and decoding settings.
- Exclude or separately label compile/cold-start time.
- Run warmup before steady-state measurement.
- Do not compare across hardware or driver versions unless the comparison explicitly aims
  to measure that difference.

## Decision Guide

| Observation | Likely cause | Action |
|---|---|---|
| Mean latency up, p95 stable | average path slower | Inspect precision, shapes and batch settings. |
| p95 up, mean stable | tail instability | Inspect outliers, concurrency and system load. |
| Throughput up, KL/PPL regressed | optimization changed quality | Treat as quality regression until explained. |
| Generation latency up, forward stable | decode/cache/tokenization overhead | Inspect KV-cache and stop conditions. |
| Short prompts fast, long prompts slow | context scaling issue | Review length buckets and cache behavior. |

## References

- Vijay Janapa Reddi et al., [MLPerf Inference Benchmark](https://arxiv.org/abs/1911.02549), 2019.
- OpenVINO, [Performance Benchmarks](https://docs.openvino.ai/2024/about-openvino/performance-benchmarks.html).
- OpenVINO, [Benchmark Tool](https://docs.openvino.ai/nightly/get-started/learn-openvino/openvino-samples/benchmark-tool.html).
- OpenVINO GenAI, [PerfMetrics API](https://docs.openvino.ai/2025/api/genai_api/_autosummary/openvino_genai.PerfMetrics.html).
