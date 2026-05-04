# Changelog

See the root `CHANGELOG.md` for release history.

The project follows human-curated release notes. Generated GitHub notes are draft
input, not the final release narrative.

## Entry policy

Each release section should group changes by user-visible impact:

- Added
- Changed
- Fixed
- Removed
- Security

Mention schema, CLI and report bundle changes explicitly. A report contract
change can affect downstream automation even when Python APIs are still
pre-release.

## Unreleased changes

During pre-release development, keep notable changes under an `Unreleased`
section until a tag is cut. Link to the matching release notes when the release
is published.

## Generated notes

GitHub-generated release notes can identify merged pull requests and
contributors. Treat them as input for a curated changelog entry, not as the final
upgrade guide.
