# Add an analysis rule

Analysis rules consume normalized observations and report context. They should
not inspect runner-specific raw outputs.

## Checklist

1. Define the evidence paths.
2. Emit a finding with severity, impact and recommendation.
3. Add snapshot tests for Markdown and HTML rendering.
4. Document the rule in the reporting reference.

## Inputs

Rules should consume normalized metrics, gate results, baseline deltas and report
metadata. If a rule needs raw runner output, normalize that evidence first in the
application/reporting layer.

## Finding requirements

Every finding should include:

- stable `id`;
- short title;
- severity;
- category;
- status;
- evidence paths;
- impact;
- recommendation;
- structured details for renderer-specific views.

## Evidence paths

Evidence paths should match report fields or normalized metric paths. For
example, use `summary.overall_status` or `likelihood.perplexity`, not a prose
description.

## Renderer interaction

Renderers may group and format findings. They must not create findings or
rewrite status. Add renderer snapshots only after the analysis rule output is
stable.
