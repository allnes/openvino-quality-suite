# Security Policy

## Supported Versions

OVIQS is currently pre-1.0. Security fixes are handled on the `main` branch until stable
release branches are introduced.

## Reporting a Vulnerability

Do not open a public issue for suspected vulnerabilities, exposed credentials, private
datasets or unsafe generated artifacts.

Use GitHub private vulnerability reporting when available for this repository. If private
reporting is not available, contact the repository maintainers through a private channel
and include:

- affected commit or version;
- reproduction steps;
- impact assessment;
- whether any credential, private model artifact or non-public dataset may be exposed.

## Publication Hygiene

Before contributing or publishing validation results:

- do not commit `.env` files, credentials, tokens, private keys or proxy settings;
- do not commit generated `reports/`, `models/`, `data/`, `datasets/`, `logs/` or cache
  directories;
- do not commit local usernames, hostnames, cloud-drive paths or internal machine
  inventories;
- sanitize benchmark reports before sharing externally;
- keep large model artifacts in an artifact store, model hub or release asset, not in Git.
