# Generation Metrics

Generation metrics evaluate text produced by a model or serving endpoint after decoding.
They are later-stage metrics: use likelihood and distribution drift first when you need to
know whether the model distribution changed, then use generation metrics to verify that
the visible output still satisfies product and task constraints.

Use them to answer:

- Does deterministic decoding still produce valid structured output?
- Did a model/runtime change introduce repetition or premature loops?
- Does the output contain required sections and avoid forbidden sections?
- Should a regression be fixed in decoding, prompt templates, schema validation or model
  quality?

## Scientific Basis

Text generation metrics generally fall into three families:

- surface-form overlap, such as n-gram overlap and repetition;
- structural validity, such as JSON parsing and schema validation;
- semantic or judge-backed quality, such as BERTScore, task-specific rubrics or LLM judges.

OVIQS currently implements deterministic rule-based generation checks. They are designed
for CI and release gates, not for broad human-preference evaluation. For open-ended prose,
pair these checks with task labels or external evaluators.

## OVIQS Implementation

Source: `src/oviqs/metrics/generation.py`.

Current built-in functions:

- `ngram_repetition_rate(text, n=3)`;
- `json_validity(text)`.

The report/playbook also reserves documented metric names for external or suite-level
checks:

- `schema_validity`;
- `required_section_coverage`;
- `forbidden_section_violation`.

Those external checks should be produced by a runner, assertion suite or integration and
then included in OVIQS reports.

## Metrics

### repetition_rate

Formula for n-grams of length `n`:

```text
total_ngrams = max(len(tokens) - n + 1, 0)
repeated = sum_g max(count(g) - 1, 0)
repetition_rate = repeated / total_ngrams
```

Direction: lower is better.

Interpretation:

- `0.0` means no repeated n-gram in the generated text.
- High values indicate loops, low-diversity decoding, cache issues, stop-sequence failure
  or prompt echo.
- Compare only with the same tokenizer/splitting rule and same `n`.

Action:

- Check greedy vs sampling settings.
- Check `temperature`, `top_p`, `top_k`, repetition penalty and stop sequences.
- If repetition appears only on OpenVINO cached generation, run serving `kv_cache_drift`.

### unique_ngram_ratio

Formula:

```text
unique_ngram_ratio = number_of_unique_ngrams / total_ngrams
```

Direction: higher is better.

Interpretation:

- Companion to `repetition_rate`.
- Short outputs can trivially score high because there are few n-grams; always inspect
  `total_ngrams`.
- Very high uniqueness does not guarantee correctness; it only says the output is not
  repeating under the chosen n-gram size.

### json_valid

Formula:

```text
json_valid = true if json.loads(text) succeeds else false
```

Direction: `true` is pass.

Interpretation:

- This is a syntactic check only.
- Valid JSON can still violate the expected schema.
- Invalid JSON means semantic evaluation should usually be skipped because downstream
  consumers cannot parse the output.

Action:

- Tighten prompt format.
- Use deterministic decoding for structured tasks.
- Add schema-level validation after syntax passes.

### schema_validity

Definition:

```text
schema_validity = valid_outputs / checked_outputs
```

or a boolean per sample when a JSON Schema/Pydantic/dataclass validator is used.

Direction: higher or `true` is better.

Interpretation:

- Checks fields, types, enums, required properties and forbidden properties.
- More useful than `json_valid` for extraction, routing, tool arguments and API responses.
- Failures usually point to prompt contract drift or insufficient constrained decoding.

Action:

- Keep schema version in the dataset or report metadata.
- Return sample-level validation errors, not only an aggregate rate.
- Gate required production schemas before semantic metrics.

### required_section_coverage

Formula:

```text
required_section_coverage =
  matched_required_sections / total_required_sections
```

Direction: higher is better.

Interpretation:

- Useful for reports, summaries, issue triage and multi-field answers.
- Does not measure content correctness inside each section.
- Low coverage means the prompt/template does not force all expected fields or decoding
  stopped early.

### forbidden_section_violation

Formula:

```text
forbidden_section_violation =
  true if any forbidden pattern/section appears else false
```

Direction: lower or `false` is better.

Interpretation:

- Use for banned boilerplate, internal-only fields, unsupported markdown sections, policy
  labels or debug text.
- Keep patterns explicit and versioned to avoid hidden evaluator drift.

## Dataset Design

Minimal generated-output row:

```json
{"id":"g1","task_type":"generation","output":"{\"answer\":\"April 16\",\"source\":\"release-notes\"}"}
```

Structured task row:

```json
{
  "id": "extract1",
  "task_type": "generation",
  "prompt": "Extract release date and device as JSON.",
  "output": "{\"release_date\":\"2026-04-16\",\"device\":\"GPU\"}",
  "metadata": {
    "schema": "release_fact_v1",
    "required_sections": ["release_date", "device"],
    "forbidden_patterns": ["PLACEHOLDER", "unknown"]
  }
}
```

Recommended datasets:

- deterministic JSON prompts for CI;
- structured extraction prompts with known schemas;
- promptfoo assertion suites for product-level contracts;
- production examples with expected entities and forbidden content;
- task-specific human-labeled sets when open-ended quality matters.

Dataset rules:

- Freeze decoding settings with the report: temperature, top-p, top-k, max tokens, seed
  and stop sequences.
- Separate short structured outputs from long-form natural language outputs.
- Store raw output exactly as returned by the runtime; do not repair JSON before scoring.
- Keep sample-level errors for debugging.

## Decision Guide

| Observation | Likely cause | Action |
|---|---|---|
| Repetition high, likelihood stable | decoding or cache issue | Check decoding parameters, stop tokens and KV-cache drift. |
| JSON invalid | output contract failed | Fix prompt/decoder before semantic scoring. |
| JSON valid, schema invalid | fields/types changed | Update prompt, schema mapping or constrained decoder. |
| Required coverage low | missing sections | Add explicit required fields and increase max tokens if truncated. |
| Forbidden violation high | unwanted boilerplate or policy leakage | Add assertion tests and inspect prompt template. |

## References

- Kishore Papineni et al., [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/), ACL 2002.
- Chin-Yew Lin, [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/), ACL Workshop 2004.
- Tianyi Zhang et al., [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675), 2019.
- Ari Holtzman et al., [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751), ICLR 2020.
- JSON Schema, [Specification](https://json-schema.org/specification).
