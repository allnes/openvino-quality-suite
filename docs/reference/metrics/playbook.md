# Metric playbook

This playbook is the reviewer-facing contents page for OVIQS metrics. Use it to
decide which metric family applies, which evidence must exist, which dataset
shape is valid, and what action to take when a value changes.

OVIQS treats a metric as actionable only when the report contains the evidence
required by the metric and the metric has reference or oracle metadata. Missing
evidence is reported as `unknown`; it must not be converted into `pass`.

## Metric detail files

| Detail file | Metrics covered | Use when |
|---|---|---|
| [Likelihood metrics](details/likelihood.md) | `token_logprobs`, `mean_log_prob`, `num_tokens`, `nll`, `perplexity`, sliding and normalized PPL variants | You have aligned causal-LM likelihood evidence. |
| [Distribution drift metrics](details/distribution-drift.md) | KL, JS, entropy drift, logit cosine, top-k overlap, target rank and sensitive-token drift | You compare a reference backend with a current backend at identical token positions. |
| [Long-context metrics](details/long-context.md) | Context gain, saturation, lost-in-middle, distractor and conflict metrics | You evaluate behavior across controlled context length, position or noise variants. |
| [Generation metrics](details/generation.md) | Repetition, structure, JSON/schema, entity and topic checks | You have generated text but not full aligned logits. |
| [RAG metrics](details/rag.md) | Retrieval ranking, evidence coverage, faithfulness, answer relevance and citation metrics | You evaluate retrieved context and answer grounding. |
| [Agent metrics](details/agent.md) | Tool validity, argument correctness, state drift, grounding, task completion and recovery | You evaluate structured traces from tool-using systems. |
| [Serving metrics](details/serving.md) | Batch invariance, prefix divergence, KV-cache drift and device drift | You compare production-serving execution modes for the same prompt. |
| [Performance metrics](details/performance.md) | Latency and throughput observations | You compare runs on the same hardware and workload profile. |

## Evidence-first decision tree

1. If the backend returns aligned logits, prefer likelihood and distribution
   drift metrics. They diagnose runtime equivalence directly.
2. If only generated text is available, use generation, RAG, agent or serving
   output metrics. Do not infer KL, JS, entropy, NLL or PPL from text-only
   generations.
3. If the task depends on retrieved evidence, use RAG metrics and keep retrieval,
   answer and citation findings separate.
4. If the task depends on tool calls or workflow state, use agent metrics and
   store expected tools, state and recovery labels with the trace fixture.
5. If the question is production stability, use serving metrics before broad
   quality metrics. A serving regression can exist even when aggregate PPL is
   unchanged.
6. If the question is capacity or speed, use performance metrics only against a
   same-hardware baseline.

## Dataset requirements

| Surface | Minimum dataset fields | Recommended public datasets or fixtures |
|---|---|---|
| Likelihood | Stable sample id, prompt text or token ids, tokenizer identity, score mask. | WikiText-2 for language-model smoke tests; deterministic OVIQS JSONL fixtures for CI. |
| Distribution drift | Same sample ids, identical tokenizer, aligned reference/current logits. | Deterministic logits fixtures, OpenVINO Runtime vs HF reference exports. |
| Long context | Controlled length labels, position labels or paired clean/noisy prompts. | RULER-style synthetic retrieval tasks, HELMET-style application tasks, controlled OVIQS fixtures. |
| Generation | Prompt, generated text, expected schema or deterministic assertions. | JSON/schema fixtures, promptfoo-style assertion rows, entity-preservation fixtures. |
| RAG | Question, retrieved contexts, expected evidence, answer and citations. | Ragas-style context/answer rows, Phoenix-style RAG evaluation exports, OVIQS RAG JSONL fixtures. |
| Agent | Trace steps, tool schemas, expected tool calls, expected state and outcome. | Deterministic OVIQS trace fixtures, DeepEval-style tool and task-completion labels. |
| Serving | Paired single/batch, full/KV-cache or device outputs/logits. | Same prompt set replayed through each serving mode. |
| Performance | Prompt/workload id, input/output token counts, timing, device profile. | Same-model baseline captured on the same machine class. |

## What to do with changes

| Symptom | First action | Do not do |
|---|---|---|
| NLL/PPL rises | Check tokenizer, score mask, dataset slice and context window before blaming runtime. | Average token-level PPL values. Aggregate NLL first, then exponentiate. |
| KL/JS spikes | Inspect alignment, vocabulary, dtype, top-k changes and sensitive tokens at the same positions. | Compare logits from different tokenizers or shifted sequences. |
| Long-context score drops | Split by length, position and distractor condition to isolate capacity, retrieval or conflict failure. | Collapse all long-context rows into one generation score. |
| RAG faithfulness drops | Inspect retrieved evidence before answer scoring; low retrieval recall makes answer faithfulness ambiguous. | Treat judge scores as comparable without recording judge/model configuration. |
| Agent recovery drops | Read the failing trace around tool-error steps and verify expected tool/state labels. | Hide trace evidence behind an aggregate score only. |
| Serving drift appears | Re-run paired prompts with deterministic settings and compare single vs batch or full vs cache first. | Mix serving drift with model quality regressions. |
| Latency changes | Confirm same hardware, model artifact, prompt length and warmup policy. | Gate latency across different machines or precision profiles. |

## Scientific and framework references

- Hugging Face Transformers, [perplexity of fixed-length models](https://huggingface.co/docs/transformers/en/perplexity), for causal-LM PPL and sliding-window evaluation.
- EleutherAI, [lm-evaluation-harness model guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md), for loglikelihood-style evaluation contracts.
- SciPy, [entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html), for Shannon entropy and KL divergence semantics.
- scikit-learn, [ndcg_score](https://sklearn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html), for ranked retrieval gain.
- SentenceTransformers, [InformationRetrievalEvaluator](https://www.sbert.net/docs/package_reference/sentence_transformer/evaluation.html), for IR metrics such as MRR, nDCG, precision and recall at k.
- Ragas, [metrics](https://docs.ragas.io/en/stable/concepts/metrics/), for context precision, context recall, faithfulness and answer relevance patterns.
- DeepEval, [tool correctness](https://deepeval.com/docs/metrics-tool-correctness), [argument correctness](https://deepeval.com/docs/metrics-argument-correctness), [task completion](https://deepeval.com/docs/metrics-task-completion) and [step efficiency](https://deepeval.com/docs/metrics-step-efficiency), for agent-trace evaluation patterns.
- NVIDIA, [RULER](https://github.com/NVIDIA/RULER), and Princeton NLP, [HELMET](https://github.com/princeton-nlp/HELMET), for long-context benchmark design.
