# RAG metrics

RAG metrics separate retrieval quality, evidence coverage, answer grounding and
citation behavior. This separation matters because answer failures can be caused
by retrieval, generation or citation mapping.

## Metrics

| Metric | Definition | Direction | Required evidence |
|---|---|---|---|
| `recall_at_k` | Relevant documents retrieved in top `k` divided by all relevant documents. | Higher is better. | Relevance labels. |
| `precision_at_k` | Relevant documents in top `k` divided by `k`. | Higher is better. | Relevance labels. |
| `mrr` | Mean reciprocal rank of the first relevant document. | Higher is better. | Ranked retrieval labels. |
| `ndcg` | Normalized discounted cumulative gain with graded relevance. | Higher is better. | Ranked graded labels. |
| `rank_quality` | Generic ranking quality score from an external evaluator. | Higher is better. | Evaluator output and config. |
| `evidence_coverage` | Expected evidence strings found in retrieved context. | Higher is better. | Expected evidence list. |
| `token_waste_ratio` | Retrieved tokens not contributing useful evidence. | Lower is better. | Context token counts and evidence labels. |
| `distractor_ratio` | Retrieved contexts marked as distractors. | Lower is better. | Distractor labels. |
| `context_precision` | Retrieved contexts that contain expected evidence divided by retrieved contexts. | Higher is better. | Retrieved contexts and expected evidence. |
| `context_recall` | Expected evidence covered by retrieved contexts. | Higher is better. | Expected evidence and contexts. |
| `faithfulness` | Answer claims supported by retrieved evidence. | Higher is better. | Claims, answer and evidence or judge labels. |
| `supported_claim_ratio` | Supported claims divided by all generated claims. | Higher is better. | Claim support labels. |
| `unsupported_claim_rate` | Unsupported claims divided by all claims. | Lower is better. | Claim support labels. |
| `contradiction_rate` | Claims contradicting evidence or reference. | Lower is better. | Contradiction labels. |
| `answer_relevance` | Answer relevance to the question. | Higher is better. | Question, answer and scorer. |
| `answer_relevancy` | Ragas-style answer relevance score. | Higher is better. | Ragas scorer output. |
| `answer_relevance_lexical` | Lexical estimate of relevance. | Higher is better. | Question and answer text. |
| `citation_precision` | Produced citations that support answer claims divided by produced citations. | Higher is better. | Citation labels. |
| `citation_recall` | Required supporting citations recovered. | Higher is better. | Required citation labels. |
| `source_correctness` | Cited sources actually support the answer. | Higher or `true` is better. | Source-to-claim labels. |
| `faithfulness_rule_based` | Literal rule-based claim support estimate. | Higher is better. | Extracted claims and retrieved text. |

## Interpretation

Read retrieval metrics before answer metrics. Low `context_recall` can make a
faithfulness failure expected rather than surprising. High retrieval scores with
low faithfulness usually points to answer generation or citation logic.

## Dataset examples

Use rows with `question`, `retrieved_contexts`, `expected_evidence`,
`reference_answer`, `answer` and `citations`. Ragas-style and Phoenix-style
exports are useful when judge configuration is recorded. Small deterministic
fixtures are preferred for CI.

## Action policy

When a RAG metric fails, classify the failure as retrieval, grounding, relevance
or citation before changing gates. Never compare judge-backed scores without
recording judge model, prompt, threshold and version.
