---
name: oviqs-metric-references
description: Use when adding, reviewing, or debugging OVIQS metric reference/oracle metadata, metric detail docs, report metric_references manifests, reference comparison outputs, reference-aware gate evaluation, or the list-metric-references CLI.
---

# OVIQS Metric References

Use this skill when a metric is added to reports, gates or standard metric matrix output.

## References

Read only what is needed:

- `src/oviqs/domain/references/catalog.py` for the reference catalog.
- `docs/metrics.md` for reference/oracle policy.
- `docs/metric_playbook.md` and `docs/metric_details/*.md` for per-metric descriptions.
- `docs/reports_and_gates.md` for manifest and gate behavior.
- `src/oviqs/adapters/reporting/reference_comparison.py` for cross-report comparison output.
- `src/oviqs/application/reporting/reference_comparison_service.py` for report comparison
  assembly.
- `tests/unit/test_metric_references.py` for catalog contract tests.
- `tests/unit/test_gates.py` for reference-aware gate behavior.

## Workflow

1. Register every gateable scalar metric in `oviqs.domain.references`.
2. Provide a primary reference, concrete sources, an oracle and a degradation rule.
3. Keep report writers automatic: `write_report` should populate `metric_references`.
4. Keep gates conservative: missing values and unreferenced metrics should be `unknown`.
5. Use section or nested paths consistently for grouped metrics such as
   `serving.batch_invariance.mean_kl`.
6. Keep `docs/metric_playbook.md` as the metric contents page and put detailed metric
   prose in the matching `docs/metric_details/<family>.md` file.
7. Update reference comparison defaults when a metric should appear in cross-report tables.
8. Update docs and tests when adding metric names, families or degradation rules.
9. Keep metric docs compatible with markdownlint and public release hygiene; examples should
   be generic and should not include generated reports or local machine details.

## Commands

```bash
.venv/bin/oviq list-metric-references
.venv/bin/oviq list-metric-references --family serving --json
.venv/bin/oviq report reference-comparison --report baseline=/tmp/baseline.json --report current=/tmp/current.json --format markdown-transposed --out /tmp/reference_comparison.md
.venv/bin/pytest tests/unit/test_metric_references.py tests/unit/test_gates.py
```
