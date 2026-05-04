# Config profiles

Configuration profiles live under `configs/`:

- `base.yaml`
- `local.yaml`
- `ci.yaml`
- `gpu.yaml`
- `prod.yaml`

Profiles should not contain secrets or machine-local paths.

## What belongs in a profile

- model or model family;
- backend and device;
- dataset slice;
- suite name;
- gate file;
- report output location;
- optional baseline selection policy.

## What does not belong in a profile

Do not hide metric interpretation or renderer behavior in profile files. Metric
semantics belong in domain code and metric references. Rendered output should be
derived from reports and analysis.

## Change discipline

Profile changes can alter quality outcomes even when code does not change. Treat
them like test policy changes: document the reason, update examples if needed,
and include the docs impact in the pull request.
