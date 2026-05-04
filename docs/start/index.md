# Start here

Start with the local dummy backend. It requires no model download and produces a
valid `EvaluationReport` quickly, which makes it the best path for checking a
new checkout, CI environment or documentation example.

OVIQS has two evaluation surfaces:

- **Logits-level diagnostics** compare probability distributions, token ranks,
  likelihood and drift. Use them when reference and current backends expose
  aligned logits for the same token positions.
- **Generation-level diagnostics** evaluate produced text, RAG outputs, agent
  traces and serving behavior. Use them when full logits are not available or
  when the production endpoint is the system under test.

The first successful workflow is:

1. Install the development environment.
2. Create a one-line JSONL fixture.
3. Run `oviq eval-likelihood --backend dummy`.
4. Validate the report with `oviq report validate`.
5. Build a bundle with `oviq report build --format all`.

After that, choose the next page by task:

- Need a copy-paste path: [Quickstart](quickstart.md).
- Need environment setup: [Installation](installation.md).
- Need a guided walkthrough: [First local run](../tutorials/first-local-run.md).
- Need the contract details: [Reporting spec](../reference/reporting/reporting-spec.md).
