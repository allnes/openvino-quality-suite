# Generation metrics

Generation metrics evaluate produced text when full aligned logits are not
available or not relevant. They should describe observable output behavior, not
pretend to be likelihood or distribution metrics.

## Metrics

| Metric | Definition | Direction | Required evidence |
|---|---|---|---|
| `ngram_repetition_rate` | Repeated n-grams divided by all n-grams. | Lower is better. | Generated text and configured `n`. |
| `ngram_repetition` | Structured repetition result object. | Review child fields. | Generated text. |
| `repetition_rate` | Alias or flattened repetition rate. | Lower is better. | Generated text. |
| `unique_ngram_ratio` | Unique n-grams divided by all n-grams. | Higher is better. | Generated text. |
| `duplicate_sentence_ratio` | Repeated generated sentences divided by all sentences. | Lower is better. | Sentence segmentation. |
| `topic_drift` | Degree of movement away from requested topic. | Lower is better. | Prompt, output and topic rubric. |
| `entity_preservation_rate` | Required entities preserved in output. | Higher is better. | Expected entity list. |
| `entity_hallucination_rate` | Unsupported new entities introduced. | Lower is better. | Allowed or source-supported entities. |
| `entity_contradiction_rate` | Entity facts contradict expected facts. | Lower is better. | Entity fact labels. |
| `date_number_version_mismatch_rate` | Date, number or version mismatches. | Lower is better. | Expected literals or extracted facts. |
| `json_validity` | Structured JSON parsing result. | Valid is better. | Generated text. |
| `json_valid` | Boolean JSON parse result. | `true` is better. | Generated text. |
| `schema_validity` | Required schema checks passed. | Higher or `true` is better. | JSON schema or structured output contract. |
| `required_section_coverage` | Required sections present. | Higher is better. | Required section list. |
| `forbidden_section_violation` | Forbidden sections appeared. | `false` is better. | Forbidden section list. |
| `markdown_structure_score` | Required Markdown structure score. | Higher is better. | Markdown structure rubric. |

## Interpretation

Rule-based generation metrics are narrow by design. JSON validity says nothing
about factuality. Repetition says nothing about retrieval grounding. Keep each
metric tied to the behavior it can actually observe.

## Dataset examples

Use JSON/schema fixtures for structured output, entity-preservation fixtures for
fact-sensitive outputs and promptfoo-style assertion rows for deterministic
release checks. Store the expected schema, sections and entity lists with the
dataset, not in a local operator note.

## Action policy

When generation checks fail, preserve the sample output in the report bundle and
fix the prompt, decoder settings or model artifact based on the specific failed
assertion. If a judge or semantic scorer did not run, leave those semantic
scores absent or `unknown`.
