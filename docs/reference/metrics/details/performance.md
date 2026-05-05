# Performance metrics

Performance metrics describe measured runtime cost. They are release-relevant
only when compared against a same-hardware, same-model, same-workload baseline.

## Metrics

| Metric | Definition | Direction | Required evidence |
|---|---|---|---|
| `forward_latency_ms_mean` | Mean full-forward latency in milliseconds. | Lower is better. | Timed forward passes and device profile. |
| `forward_latency_ms_p95` | 95th percentile full-forward latency. | Lower is better. | Timed forward passes and enough samples for tail review. |
| `tokens_per_second_forward` | Forward-pass token throughput. | Higher is better. | Token count and elapsed forward time. |
| `generation_latency_ms` | End-to-end generation latency. | Lower is better. | Timed generation request and decode settings. |

## Interpretation

Latency and throughput are operational signals, not model-quality signals.
Compare them only when the hardware, driver, model artifact, precision, prompt
length distribution, batch size and warmup policy are comparable.

## Dataset examples

Use fixed prompt buckets for smoke checks and production-like prompt length
distributions for release checks. Store summarized timing observations in
reports; keep raw local profiler output and machine inventories out of source
control.

## Action policy

If a performance metric regresses, verify measurement conditions first. If the
run conditions changed, create a new baseline rather than failing a quality gate
against an incompatible baseline.
