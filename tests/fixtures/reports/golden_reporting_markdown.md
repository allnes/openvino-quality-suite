# OVIQS Report: snapshot-run

## Executive Summary

- Overall status: **WARNING**
- Model: `m`
- Reference: `cpu`
- Current: `gpu`
- Device: `GPU`
- Precision: `FP16`
- Suite: `snapshot-suite`
- Created: `2026-05-03T00:00:00Z`
- Gates: 2 pass / 1 warning / 0 fail / 0 unknown

## Report Findings

1. needs review

## Top Analysis Findings

| Severity | Status | Finding | Recommendation |
|---|---|---|---|
| medium | warning | Likelihood section is warning | Inspect tokenizer alignment, precision, model export and calibration data. |

## Gate Results

| Section | Gate | Metric Path | Value | Threshold | Status | Reference |
|---|---|---|---:|---:|---|---|
| Likelihood | `perplexity_max` | `perplexity` | 10 | 8 | warning | present |

## Diagnostic Cards

| Section | Status | Key Metrics | Thresholds | Baseline Delta | Interpretation | Details |
|---|---|---|---|---|---|---|
| Likelihood | warning | perplexity=10 | warning=1 | no baseline | 1 warning metric(s); compare against baseline and gates. | [details](#likelihood) |
| Reproducibility | pass | seed=7, oviqs=test | no gated metrics | no baseline | seed=7, oviqs=test | [details](#reproducibility) |

## Reproducibility

| Field | Value |
|---|---|
| `seed` | 7 |
| `versions.oviqs` | test |

## Metric Details

| Path | Value | Baseline | Delta | Threshold | Status | Reference | Rule |
|---|---:|---:|---:|---:|---|---|---|
| `likelihood.perplexity` | 10 | n/a | n/a | 8 | warning | lm-evaluation-harness loglikelihood/loglikelihood_rolling | Degraded when current NLL/PPL or configured delta exceeds the reference baseline gate. |
| `reproducibility.seed` | 7 | n/a | n/a | n/a | pass | n/a | n/a |
| `reproducibility.versions.oviqs` | test | n/a | n/a | n/a | pass | n/a | n/a |

## Developer Debug Appendix

### Likelihood

- Raw JSON path: `likelihood`
- Evidence paths: `likelihood.perplexity`

### Reproducibility

- Raw JSON path: `reproducibility`
- Evidence paths: `reproducibility.seed`, `reproducibility.versions.oviqs`
