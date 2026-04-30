# Repo-Local Skills

The `skills/` directory contains compact repository automation workflows for repeatable
OVIQS work. They are project assets, not runtime package code.

## Available skills

- `skills/oviqs-evaluation/SKILL.md`: run OVIQS diagnostics, reports and gates.
- `skills/oviqs-development/SKILL.md`: implement or update OVIQS code while preserving
  metric and runner boundaries.
- `skills/oviqs-gpu-metrics/SKILL.md`: run GPU metric verification, GenAI matrix
  exports and remote GPU smoke workflows.
- `skills/oviqs-integrations/SKILL.md`: work with optional evaluator adapters,
  integration requirements and external result normalization.
- `skills/oviqs-metric-references/SKILL.md`: maintain metric reference/oracle metadata,
  report manifests, metric detail docs, reference comparisons and reference-aware gates.

## Maintenance rules

- Keep each skill focused and short.
- Put only `SKILL.md` and directly useful bundled resources under a skill directory.
- Do not add generated reports, virtual environments, caches or model artifacts to skills.
- Update skills when CLI commands, report sections, runner names or validation rules change.
- Update skills when CI checks, dependency files, runner contracts or public release
  hygiene rules change.
- Keep skill examples generic and public-safe; do not encode local machine paths,
  credentials, private inventory or generated report payloads.
