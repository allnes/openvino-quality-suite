---
name: oviqs-metric-references
description: Use when adding, reviewing, or debugging OVIQS metric reference/oracle metadata, metric detail docs, report metric_references manifests, reference comparison outputs, reference-aware gate evaluation, or the list-metric-references CLI.
---

# OVIQS Metric References

Use this skill when a metric is added to reports, gates or standard metric matrix output.

## References

Read only what is needed:

- `src/oviqs/domain/references/catalog.py` for the reference catalog.
- `docs/reference/metrics/playbook.md` for reviewer-facing metric formulas, datasets and
  triage actions.
- `docs/reference/metrics/catalogue.md` for public metric paths and family semantics.
- `docs/reference/metrics/references-and-oracles.md` for reference/oracle policy.
- `docs/reference/reporting/reporting-spec.md` and `docs/reference/reporting/bundle-layout.md`
  for manifest, bundle and gate review behavior.
- `docs/reference/config/gates.md` for gate syntax and review rules.
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
6. Keep `docs/reference/metrics/playbook.md` as the public metric contents page, keep
   `docs/reference/metrics/catalogue.md` as the path contract, and update related
   reporting, gate or tutorial pages when the review workflow changes.
7. Update reference comparison defaults when a metric should appear in cross-report tables.
8. Update docs and tests when adding metric names, families or degradation rules.
9. Regenerate generated docs/examples when schema, CLI or bundle output changes.
10. Keep metric docs compatible with markdownlint and public release hygiene; examples should
    be generic and should not include generated reports or local machine details.

## Commands

```bash
.venv/bin/oviq list-metric-references
.venv/bin/oviq list-metric-references --family serving --json
.venv/bin/oviq report reference-comparison --report baseline=/tmp/baseline.json --report current=/tmp/current.json --format markdown-transposed --out /tmp/reference_comparison.md
.venv/bin/pytest tests/unit/test_metric_references.py tests/unit/test_gates.py
```
