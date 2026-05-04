# Add a renderer

Renderers consume reports and analysis outputs through reporting ports. They must
not compute metric status.

## Checklist

1. Implement the renderer under `oviqs.adapters.reporting`.
2. Register an entry point when it should be externally discoverable.
3. Add output snapshots.
4. Check accessibility for HTML output.

## Renderer contract

A renderer receives report or bundle data and returns a representation such as
Markdown or HTML. It may:

- choose grouping and ordering;
- format tables;
- link evidence paths to sections;
- include static assets.

It must not:

- compute metric values;
- change status;
- hide `unknown` findings;
- fetch network resources for a self-contained dashboard.

## HTML accessibility

HTML output should have a document title, semantic headings, readable tables and
alt text for informative images. Prefer text commands and tables over terminal
screenshots.

## Tests

Use focused snapshots for stable structure. Avoid snapshots that fail on
irrelevant timestamps or ordering unless those fields are part of the contract.
