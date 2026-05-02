# RAG Metrics

RAG metrics evaluate retrieval and answer grounding separately. This separation is
important: a generated answer can be wrong because retrieval missed evidence, retrieval
returned too many distractors, the model ignored correct evidence, or citations were not
attached to the supported claims.

Use them to answer:

- Did retrieval bring the expected evidence into context?
- Is retrieved context precise enough, or does it contain too much noise?
- Are citations aligned with expected sources?
- Are final answer claims grounded in retrieved context?

## Scientific Basis

Retrieval-augmented generation combines a retriever and a generator. Retrieval quality is
usually measured with information-retrieval metrics such as precision, recall, MRR and
nDCG. Answer grounding is often measured with claim support, citation correctness or
judge-backed faithfulness. OVIQS currently provides deterministic rule-based RAG checks and
integration hooks for judge-backed evaluators.

OVIQS built-in RAG metrics use exact or substring matching. They are conservative and
reproducible, which makes them good for CI, but they do not detect paraphrases unless an
external scorer is configured.

## OVIQS Implementation

Source: `src/oviqs/domain/metrics/rag.py`.

Core functions:

- `evidence_coverage(expected_evidence, retrieved_contexts)`;
- `context_precision(expected_evidence, retrieved_contexts)`;
- `context_recall(expected_evidence, retrieved_contexts)`;
- `citation_metrics(expected_citations, actual_citations)`;
- `rule_based_faithfulness(answer, retrieved_contexts, claims)`;
- `distractor_ratio(retrieved_contexts, relevant_context_indices)`;
- `supported_claim_ratio_placeholder()` for judge-backed scoring.

## Metrics

### evidence_coverage

Formula:

```text
evidence_coverage =
  matched_expected_evidence_strings / total_expected_evidence_strings
```

Direction: higher is better.

Interpretation:

- `1.0` means every expected evidence string appears in retrieved contexts.
- Low coverage means retrieval failed before generation.
- Because matching is substring-based and lowercased, paraphrased evidence can be missed.

Action:

- Check corpus indexing and chunk boundaries.
- Increase top-k or improve query rewriting if evidence is absent.
- Add aliases/synonyms to labels only when they are legitimate answer variants.

### context_precision

Formula:

```text
context_precision =
  retrieved_contexts_containing_expected_evidence / total_retrieved_contexts
```

Direction: higher is better.

Interpretation:

- Measures how much retrieved context is useful under exact evidence labels.
- Low precision means the prompt is carrying distractors.
- High precision does not guarantee recall; the retriever may return only one of several
  required facts.

Action:

- Reduce top-k when recall is already high.
- Add reranking or chunk filtering.
- Adjust chunk size to avoid mixing relevant and irrelevant material.

### context_recall

Formula:

```text
context_recall =
  matched_expected_evidence_strings / total_expected_evidence_strings
```

Direction: higher is better.

In the current OVIQS implementation, `context_recall` is the same evidence-string coverage
calculation as `evidence_coverage`, exposed under retrieval terminology.

Interpretation:

- Low recall means important evidence is missing.
- Recall is usually the first RAG gate: if evidence is not retrieved, answer faithfulness
  cannot be expected.

### citation_precision

Formula:

```text
citation_precision =
  |expected_citations intersect actual_citations| / |actual_citations|
```

Direction: higher is better.

Interpretation:

- Low precision means the answer cites irrelevant or unsupported sources.
- If there are no actual citations, OVIQS divides by `max(len(actual), 1)`, so precision is
  `0.0` unless there are no expected citations and the dataset defines that as a pass
  externally.

Action:

- Ensure citation IDs in generated answers use the same canonical IDs as retrieval labels.
- Penalize hallucinated citations before judging answer text.

### citation_recall

Formula:

```text
citation_recall =
  |expected_citations intersect actual_citations| / |expected_citations|
```

Direction: higher is better.

Interpretation:

- Low recall means expected sources were not cited.
- The answer may be textually correct but unsupported for audit purposes.

### faithfulness

Formula:

```text
faithfulness = supported_claims / total_claims
```

OVIQS rule-based implementation counts a claim as supported when the lowercased claim text
appears literally in the answer or retrieved contexts.

Direction: higher is better.

Interpretation:

- Good for deterministic claims such as exact dates, IDs, names and short facts.
- Conservative for paraphrases and multi-sentence reasoning.
- Do not use as the only gate for natural-language answers unless claims are extracted
  into exact strings.

Action:

- Add claim extraction if answers are long.
- Add a judge-backed scorer for paraphrase support.
- Review unsupported claims before changing retrieval.

### distractor_ratio

Formula:

```text
distractor_ratio =
  (total_retrieved_contexts - relevant_contexts) / total_retrieved_contexts
```

Direction: lower is better.

Interpretation:

- High distractor ratio means the generator receives many irrelevant chunks.
- High distractor ratio can cause long-context degradation and wrong-source answers even
  when recall is acceptable.

Action:

- Tune top-k, reranking and chunk size.
- Add source-type filters.
- Move authoritative evidence closer to prompt edges if long-context metrics show
  lost-in-the-middle behavior.

### supported_claim_ratio

Definition: judge-backed or external scorer estimate of claim support.

Direction: higher is better.

OVIQS built-in function returns `unknown`/placeholder unless an external scorer is wired.
Use it when exact substring faithfulness is too strict.

Implementation guidance:

- Keep judge model name, prompt, rubric and version in report metadata.
- Calibrate judge-backed scores against a small human-labeled set.
- Do not compare scores across judge models without recalibration.

## Dataset Design

RAG dataset row:

```json
{
  "id": "r1",
  "task_type": "rag",
  "prompt": "When was the release?",
  "retrieved_contexts": [
    "Release notes: version 1.2 shipped on April 16, 2026.",
    "Unrelated roadmap note."
  ],
  "expected_evidence": ["April 16, 2026"],
  "references": ["release-notes-v1.2"],
  "metadata": {"relevant_context_indices": [0]}
}
```

Answer row:

```json
{
  "id": "r1",
  "answer": "The release was April 16, 2026.",
  "claims": ["April 16, 2026"],
  "citations": ["release-notes-v1.2"]
}
```

Recommended datasets:

- small hand-labeled retrieval fixtures for CI;
- SQuAD-style question/context/answer rows;
- Natural Questions style open-domain QA rows;
- production RAG logs with known relevant document IDs;
- MTEB retrieval tasks for embedding/retriever benchmarking;
- RAGAS/Phoenix datasets when judge-backed scoring is configured.

Dataset rules:

- Canonicalize source IDs before scoring citations.
- Keep expected evidence short, exact and auditable.
- Store retrieved contexts in the same order used by the generator.
- Include negative/distractor contexts, not only clean positives.
- Separate retrieval-only regressions from answer-generation regressions.

## Decision Guide

| Observation | Likely cause | Action |
|---|---|---|
| Context recall low | evidence missing | Fix retriever, query rewrite, indexing or top-k. |
| Recall high, precision low | too much noise | Add reranker, reduce top-k or change chunking. |
| Citations low, faithfulness high | answer right but support omitted | Fix citation formatting and source ID mapping. |
| Faithfulness low, recall high | generator ignored evidence | Adjust prompt, context ordering or model. |
| Distractor ratio high and long-context score poor | retrieval noise hurts context use | Tune retrieval and reranking before model export changes. |
| Supported claim ratio unknown | scorer absent | Add judge-backed scorer or keep metric out of gates. |

## References

- Patrick Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), 2020.
- Shahul Es et al., [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217), 2023.
- scikit-learn, [nDCG score](https://sklearn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html).
- Nils Reimers and Iryna Gurevych, [Sentence-BERT](https://arxiv.org/abs/1908.10084), 2019.
- Pranav Rajpurkar et al., [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250), 2016.
- Tom Kwiatkowski et al., [Natural Questions: A Benchmark for Question Answering Research](https://aclanthology.org/Q19-1026/), TACL 2019.
- Niklas Muennighoff et al., [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316), 2022.
