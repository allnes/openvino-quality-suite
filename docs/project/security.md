# Security

See the root `SECURITY.md` for vulnerability reporting and publication hygiene.

Do not publish credentials, private datasets, private model artifacts or
unsanitized machine-local report output.

## Report hygiene

Reports and bundles can contain model paths, dataset paths, prompts, generated
text and machine-local metadata. Before attaching them to a public issue or pull
request:

1. Remove secrets and private paths.
2. Check that prompts and outputs are shareable.
3. Prefer minimal reproducer fixtures.
4. Keep private model artifacts out of the repository.

## Dependency hygiene

The full pre-commit gate runs dependency and security checks. Do not silence
audit findings without documenting the reason and the planned follow-up.

## Disclosure

Use the root security policy for vulnerability reports. Do not disclose
exploitable behavior in public issues before maintainers have had time to assess
the report.
