# Analysis model

Analysis findings are human-readable conclusions derived from metrics, gates,
baselines and references.

Each finding should include:

- severity
- category
- status
- evidence paths
- impact
- recommendation

Renderer logic should display findings; it should not create them.

## Finding shape

A finding should be specific enough to act on:

```json
{
  "id": "unknown-overall-status",
  "title": "Overall status is unknown",
  "severity": "medium",
  "category": "summary",
  "status": "unknown",
  "evidence_paths": ["summary.overall_status"],
  "impact": "The report cannot be used as a quality gate without more evidence.",
  "recommendation": "Inspect missing metrics and configure references or gates.",
  "details": {}
}
```

## Severity

Use severity to rank reviewer attention:

| Severity | Use when |
|---|---|
| `low` | The issue is informational or affects a narrow optional path. |
| `medium` | A gate or important metric is inconclusive. |
| `high` | A core metric is degraded or a required check failed. |
| `critical` | The report indicates a release-blocking quality problem. |

## Evidence paths

Evidence paths should point back to normalized metric paths or report fields.
They let reviewers move from a finding to the raw data without guessing.

## Renderer contract

Renderers may sort, group and format findings. They must not create new
findings, change statuses or hide `unknown` evidence.
