# Long-context metrics

Long-context metrics isolate whether additional context helps, stops helping or
hurts under controlled length, position, distractor and conflict conditions.

## Metrics

| Metric | Definition | Direction | Required evidence |
|---|---|---|---|
| `context_gain` | Baseline NLL minus long-context NLL. | Higher is better. | Matched short and long variants. |
| `context_gain_64k` | Context gain measured at the 64k-token condition. | Higher is better. | 64k controlled condition. |
| `nll_by_context_length` | NLL curve over ordered context lengths. | Lower is better at each length. | Length-labeled runs. |
| `context_saturation_curve` | NLL, PPL and gain by length. | Higher gain until saturation. | Length-labeled runs. |
| `lost_in_middle_score` | Middle-position degradation versus edge positions. | Lower is better. | Position-bucketed PPL or quality. |
| `degradation_slope` | Slope of quality or NLL over log-scaled context length. | Lower degradation is better. | At least two lengths. |
| `degradation_slope_quality` | Higher-is-better quality slope over length. | Near zero or positive is better. | Quality by context length. |
| `distractor_sensitivity` | `distracted_nll - clean_nll`. | Lower is better. | Paired clean and distracted prompts. |
| `clean_nll` | NLL without distractors. | Lower is better. | Clean context prompt. |
| `distracted_nll` | NLL with distractors. | Lower is better; compare to clean. | Paired distracted prompt. |
| `faithfulness_drop` | Drop in faithfulness caused by long/noisy context. | Lower is better. | Answer and evidence labels. |
| `supported_claim_ratio_drop` | Drop in supported claims under perturbation. | Lower is better. | Claim support labels. |
| `context_gain_drop` | Loss of gain under noise or conflict. | Lower is better. | Clean/noisy gain pair. |
| `entropy_shift_with_distractors` | Confidence change caused by distractors. | Near zero is best unless expected. | Logits for clean/noisy pair. |
| `authoritative_margin` | Authoritative answer log-prob minus best conflicting answer log-prob. | Higher is better. | Candidate answer log-probs. |
| `candidate_logprobs` | Log probabilities assigned to candidate answers. | Context metric. | Candidate answer set. |
| `conflict_nll` | NLL on conflict-resolution examples. | Lower is better. | Conflict-labeled prompt. |
| `conflict_sensitivity` | NLL or quality change caused by conflict. | Lower degradation is better. | Clean/conflicted pair. |
| `conflict_entropy` | Entropy across conflicting candidate answers. | Lower can mean decisive; inspect with correctness. | Candidate log-probs. |
| `source_mixup_rate` | Rate of choosing an incorrect or unsupported source. | Lower is better. | Source labels. |
| `unsupported_resolution_rate` | Rate of conflict resolutions without evidence support. | Lower is better. | Evidence support labels. |
| `conflict_contradiction_rate` | Contradiction rate on conflict examples. | Lower is better. | Contradiction labels. |

## Interpretation

Long context is not one metric. Split regressions by length, position and
perturbation type. A model can improve on `context_gain` while failing
`lost_in_middle_score`, or remain stable on clean prompts while failing
`distractor_sensitivity`.

## Dataset examples

Use RULER-style synthetic tasks for controlled length and position stress. Use
HELMET-style application tasks when the goal is end-to-end long-context utility.
For CI, use small paired fixtures with explicit clean, distracted and conflict
labels.

## Action policy

If only one length or one prompt variant was run, mark length-curve and paired
perturbation metrics `unknown`. When a long-context metric fails, inspect the
paired sample and the position bucket before changing a global quality gate.
