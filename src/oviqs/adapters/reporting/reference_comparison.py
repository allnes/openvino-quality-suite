from __future__ import annotations

import csv
import html
import io
import json
from typing import Any

SECTION_DESCRIPTIONS = {
    "likelihood": (
        "Language-model likelihood on standard text. Lower NLL/PPL means the model assigns "
        "higher probability to the reference corpus."
    ),
    "inference_equivalence": (
        "Runtime numerical equivalence. These metrics compare logits distributions across "
        "devices or execution variants."
    ),
    "long_context": (
        "Controlled context-use diagnostics: context gain, position sensitivity, distractors "
        "and conflicting evidence."
    ),
    "generation": (
        "Application-level output checks: structure, repetition, entity preservation and "
        "contract compliance."
    ),
    "serving": (
        "Serving correctness under batching and cached decode. The same prompt should keep "
        "stable logits and generated prefixes."
    ),
    "rag": ("Retrieval and grounding checks with labeled evidence, citations and answer support."),
    "agent": (
        "Trace-level checks for tool calls, state, recovery, grounding and policy constraints."
    ),
    "performance": "Same-hardware latency and throughput measurements.",
}

SECTION_ORDER = (
    "likelihood",
    "inference_equivalence",
    "long_context",
    "generation",
    "serving",
    "rag",
    "agent",
    "performance",
)

METRIC_EXPLANATIONS = {
    "token_logprobs": "Per-token log probabilities assigned to target tokens in teacher forcing.",
    "mean_log_prob": "Average target-token log probability; this is the negative of NLL.",
    "num_tokens": "Number of tokens included in the metric denominator.",
    "nll": "Mean negative log-likelihood. Lower is better.",
    "nll_delta_vs_ref": (
        "NLL change against a reference run for the same model, data and tokenizer."
    ),
    "perplexity": "exp(NLL). Lower means better corpus prediction.",
    "ppl_relative_delta_vs_ref": "Relative PPL change against a reference run.",
    "sliding_window_ppl": "Perplexity computed over overlapping fixed-size context windows.",
    "word_perplexity": "Perplexity normalized by word count for tokenizer-sensitive comparisons.",
    "byte_perplexity": "Perplexity normalized by byte count for tokenizer-sensitive comparisons.",
    "bits_per_byte": "Byte-normalized language-model score. Lower is better.",
    "length_bucketed_ppl": "Perplexity grouped by total sample length buckets.",
    "effective_context_bucketed_ppl": "Perplexity grouped by available left-context buckets.",
    "position_bucketed_ppl": "Perplexity grouped by relative token position in the prompt.",
    "kl_per_pos": "KL divergence for each aligned token position.",
    "mean_kl": "Mean KL divergence between reference and current token distributions.",
    "p95_kl": "95th percentile KL; highlights tail runtime drift.",
    "max_kl": "Maximum KL divergence over aligned token positions.",
    "js_per_pos": "Jensen-Shannon divergence for each aligned token position.",
    "mean_js": "Mean Jensen-Shannon divergence between token distributions.",
    "p95_js": "95th percentile Jensen-Shannon divergence.",
    "ref_entropy_per_pos": "Reference distribution entropy at each token position.",
    "cur_entropy_per_pos": "Current distribution entropy at each token position.",
    "entropy_drift_per_pos": "Per-position entropy difference between current and reference.",
    "mean_entropy_drift": "Mean entropy difference between current and reference distributions.",
    "logit_cosine_per_pos": "Cosine similarity of logit vectors at each token position.",
    "mean_logit_cosine": "Mean cosine similarity over aligned logit vectors.",
    "top1_changed_rate": "Fraction of positions where the top-1 token changed.",
    "top5_overlap": "Average overlap of top-5 token sets.",
    "top10_overlap": "Average overlap of top-10 token sets.",
    "topk_overlap": "Average overlap of top-k token sets.",
    "target_rank_delta": "Change in target-token rank between two distributions.",
    "sensitive_token_drift": "Distribution drift on a configured sensitive token set.",
    "context_gain": "NLL improvement from providing longer or richer context.",
    "context_gain_64k": "Context gain measured at the 64k-token condition.",
    "nll_by_context_length": "NLL curve across controlled context lengths.",
    "context_saturation_curve": "Quality or NLL curve showing where extra context stops helping.",
    "schema_validity": "1 means required structure/schema checks passed.",
    "lost_in_middle_score": "Positive means middle context is worse than edges.",
    "degradation_slope": "Slope of quality or NLL as context length increases.",
    "degradation_slope_quality": "Higher-is-better quality slope across context lengths.",
    "distractor_sensitivity": "NLL change caused by distractor context.",
    "clean_nll": "NLL on a clean context without distractors.",
    "distracted_nll": "NLL on the paired context with distractors.",
    "faithfulness_drop": "Drop in answer faithfulness caused by long/noisy context.",
    "supported_claim_ratio_drop": "Drop in supported-claim ratio caused by distractors.",
    "context_gain_drop": "Loss of context gain under noise or conflict.",
    "entropy_shift_with_distractors": "Entropy change caused by distractor context.",
    "authoritative_margin": "Authoritative answer log-prob minus best conflicting answer log-prob.",
    "candidate_logprobs": "Log probabilities assigned to candidate answers in a conflict task.",
    "conflict_nll": "NLL on conflict-resolution examples.",
    "conflict_sensitivity": "Quality or NLL sensitivity to conflicting evidence.",
    "conflict_entropy": "Entropy over conflicting candidate answers.",
    "source_mixup_rate": "Rate of choosing the wrong or unsupported source.",
    "unsupported_resolution_rate": "Rate of answers resolving conflict without evidence support.",
    "contradiction_rate": "Fraction of contradiction/conflict failures.",
    "conflict_contradiction_rate": "Contradiction rate on explicit conflict-resolution cases.",
    "ngram_repetition": "Summary object for repeated n-gram behavior.",
    "ngram_repetition_rate": "Repeated n-gram fraction. Lower is better.",
    "repetition_rate": "Repeated n-gram fraction. Lower is better.",
    "unique_ngram_ratio": "Unique n-grams divided by all n-grams. Higher is better.",
    "duplicate_sentence_ratio": "Fraction of generated sentences repeated verbatim.",
    "topic_drift": "Degree to which generation moves away from the requested topic.",
    "entity_preservation_rate": "Fraction of required entities preserved in generation.",
    "entity_hallucination_rate": "Rate of unsupported new entities in generation.",
    "entity_contradiction_rate": "Rate of entity claims contradicting expected facts.",
    "date_number_version_mismatch_rate": "Rate of date, number or version mismatches.",
    "json_validity": "Structured JSON parsing result object.",
    "json_valid": "Boolean JSON parse result; false means invalid JSON text.",
    "required_section_coverage": "Fraction of required output sections present.",
    "forbidden_section_violation": "Whether forbidden output sections appeared.",
    "markdown_structure_score": "Score for required Markdown structure.",
    "batch_invariance": "Summary of same-sample single-vs-batch serving drift.",
    "batch_invariance_drift": "Distribution drift caused by batching.",
    "batch_invariance_mean_kl": "Mean KL for single-sample vs batched logits.",
    "batch_mean_kl": "Mean drift between single-sample and batched inference.",
    "batch_p95_kl": "Tail drift between single-sample and batched inference.",
    "batch_js": "Mean JS divergence between single-sample and batched logits.",
    "batch_entropy_drift": "Mean entropy shift introduced by batching.",
    "batch_top1_changed_rate": "Top-1 instability introduced by batching.",
    "batch_generation_prefix_divergence": "Prefix divergence for single-vs-batch generation.",
    "generation_prefix_divergence": "Prefix divergence between two deterministic generations.",
    "prefix_divergence_rate": "Normalized share of generated prefix that diverged.",
    "kv_cache_drift": "Summary of full-forward vs KV-cache decode drift.",
    "kv_cache_mean_kl": "Mean KL for full-forward vs cached decode logits.",
    "kv_cache_p95_kl": "P95 KL for full-forward vs cached decode logits.",
    "kv_mean_kl": "Mean drift between full-forward logits and cached decode logits.",
    "kv_p95_kl": "P95 drift between full-forward logits and cached decode logits.",
    "kv_mean_js": "Mean JS divergence for full-forward vs cached decode logits.",
    "kv_entropy_drift": "Mean entropy shift introduced by cached decode.",
    "kv_top1_change_rate": "Top-1 instability in cached decode.",
    "kv_generation_divergence": "Generated text divergence between full and cached decode.",
    "device_drift": "Distribution drift between devices or precision variants.",
    "recall_at_k": "Fraction of relevant documents retrieved in top-k.",
    "precision_at_k": "Fraction of top-k retrieved documents that are relevant.",
    "mrr": "Mean reciprocal rank of the first relevant retrieved document.",
    "ndcg": "Ranking quality with graded relevance labels.",
    "rank_quality": "Generic retriever or reranker quality score.",
    "evidence_coverage": "Fraction of expected evidence strings present in retrieved context.",
    "token_waste_ratio": "Share of retrieved tokens not contributing useful evidence.",
    "distractor_ratio": "Share of retrieved contexts marked as distractors.",
    "context_precision": "Fraction of retrieved contexts that are relevant.",
    "context_recall": "Fraction of expected evidence covered by retrieved contexts.",
    "faithfulness": "Fraction of answer claims supported by retrieved evidence.",
    "supported_claim_ratio": "Generated claims supported by evidence divided by all claims.",
    "unsupported_claim_rate": "Generated claim rate not supported by retrieved evidence.",
    "answer_relevance": "Answer relevance to the question.",
    "answer_relevancy": "Ragas-style answer relevance score.",
    "answer_relevance_lexical": "Lexical answer relevance estimate.",
    "citation_precision": "Fraction of produced citations that support the answer.",
    "citation_recall": "Fraction of required citations recovered by the answer.",
    "source_correctness": "Whether cited sources actually support the answer.",
    "faithfulness_rule_based": "Literal rule-based claim support estimate.",
    "tool_call_validity": "Fraction of tool calls satisfying tool schema/policy checks.",
    "tool_correctness": "Whether expected tools were called correctly.",
    "argument_correctness": "Semantic correctness of tool-call arguments.",
    "redundant_tool_call_rate": "Repeated or unnecessary tool calls divided by all tool calls.",
    "step_efficiency": "Trace efficiency compared with an expected or minimal trajectory.",
    "agent_state_drift": "Mismatch between expected and actual agent state.",
    "state_drift_score": "Fraction of expected state fields with wrong values.",
    "observation_grounding_score": "Final claims grounded in tool observations.",
    "task_completion": "Whether the trace reached the expected final outcome.",
    "policy_violation_rate": "Share of trace steps marked as policy violations.",
    "recovery_score": "Fraction of tool-error scenarios recovered successfully.",
    "recovery_after_tool_error": "Share of tool errors followed by a recovery action.",
    "same_error_repeat_rate": "Rate of repeating the same tool error.",
    "fallback_quality_score": "Quality of fallback behavior after tool errors.",
    "unsafe_recovery_rate": "Rate of unsafe recovery behavior after tool errors.",
    "unnecessary_user_clarification_rate": "Rate of avoidable clarification turns.",
    "forward_latency_ms_mean": "Mean full-forward latency on the measured hardware.",
    "forward_latency_ms_p95": "P95 full-forward latency on the measured hardware.",
    "tokens_per_second_forward": "Forward-pass token throughput.",
    "generation_latency_ms": "End-to-end generation latency on the measured hardware.",
}


def render_reference_comparison_markdown(comparison: dict[str, Any]) -> str:
    labels = [item["label"] for item in comparison["reports"]]
    headers = [
        "Section",
        "Metric",
        "Reference",
        "Degradation rule",
        *labels,
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in comparison["rows"]:
        values = [
            row["section"],
            row["metric"],
            row.get("reference", ""),
            row.get("degradation_rule", ""),
            *[status_value(row, label) for label in labels],
        ]
        lines.append("| " + " | ".join(_escape_md(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def render_transposed_reference_comparison_markdown(comparison: dict[str, Any]) -> str:
    labels = [item["label"] for item in comparison["reports"]]
    metric_headers = [f"{row['section']}.{row['metric']}" for row in comparison["rows"]]
    lines = [
        "# Metric Values",
        "",
        "| Model | " + " | ".join(metric_headers) + " |",
        "|" + "|".join(["---"] * (len(metric_headers) + 1)) + "|",
    ]
    for label in labels:
        values = [status_value(row, label) for row in comparison["rows"]]
        lines.append("| " + " | ".join(_escape_md(value) for value in [label, *values]) + " |")

    lines.extend(
        [
            "",
            "# Metric References",
            "",
            "| Metric | Reference | Degradation rule |",
            "|---|---|---|",
        ]
    )
    for row in comparison["rows"]:
        metric = f"{row['section']}.{row['metric']}"
        lines.append(
            "| "
            + " | ".join(
                _escape_md(value)
                for value in [
                    metric,
                    row.get("reference", ""),
                    row.get("degradation_rule", ""),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_reference_comparison(
    comparison: dict[str, Any],
    output_format: str,
) -> str:
    if output_format == "json":
        return json.dumps(comparison, indent=2, ensure_ascii=False) + "\n"
    if output_format == "markdown":
        return render_reference_comparison_markdown(comparison)
    if output_format == "markdown-transposed":
        return render_transposed_reference_comparison_markdown(comparison)
    if output_format in {"html-dashboard", "html-by-model"}:
        return render_reference_comparison_html_by_model(comparison)
    if output_format == "csv":
        labels = [item["label"] for item in comparison["reports"]]
        fieldnames = ["section", "metric", "reference", "degradation_rule", *labels]
        rows = [
            {
                "section": row["section"],
                "metric": row["metric"],
                "reference": row.get("reference", ""),
                "degradation_rule": row.get("degradation_rule", ""),
                **{label: status_value(row, label) for label in labels},
            }
            for row in comparison["rows"]
        ]
        pandas_csv = _render_csv_with_pandas(rows, fieldnames)
        if pandas_csv is not None:
            return pandas_csv
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue()
    raise ValueError(f"Unsupported reference comparison format: {output_format}")


def _render_csv_with_pandas(rows: list[dict[str, Any]], fieldnames: list[str]) -> str | None:
    try:
        import pandas as pd
    except ImportError:
        return None
    return pd.DataFrame(rows, columns=fieldnames).to_csv(index=False)


def render_reference_comparison_html_dashboard(comparison: dict[str, Any]) -> str:
    return render_reference_comparison_html_by_model(comparison)


def render_reference_comparison_html_by_model(comparison: dict[str, Any]) -> str:
    labels = [item["label"] for item in comparison["reports"]]
    report_paths = {item["label"]: item.get("path", "") for item in comparison["reports"]}
    rows_by_section = _rows_by_section(comparison["rows"])
    ordered_sections = [section for section in SECTION_ORDER if section in rows_by_section]
    ordered_sections.extend(
        section for section in rows_by_section if section not in ordered_sections
    )
    model_buttons = "\n".join(
        _model_button(label, comparison["rows"], active=index == 0)
        for index, label in enumerate(labels)
    )
    model_panels = "\n".join(
        _model_panel(
            label,
            comparison["rows"],
            ordered_sections,
            report_path=str(report_paths.get(label, "")),
            active=index == 0,
        )
        for index, label in enumerate(labels)
    )
    critical_rows = "\n".join(
        _critical_metric_row(row, labels) for row in _critical_metric_rows(comparison["rows"])
    )
    critical_headers = "\n".join(f"<th>{_h(label)}</th>" for label in labels)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OVIQS Target Model Quality Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --panel-soft: #f1f4f8;
      --text: #16202a;
      --muted: #5d6b7a;
      --line: #d8dee8;
      --accent: #0068b5;
      --bad: #b42318;
      --warn: #b54708;
      --good: #067647;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --sans: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      line-height: 1.45;
    }}
    header {{ border-bottom: 1px solid var(--line); background: var(--panel); }}
    .topbar {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 16px 24px 12px;
    }}
    h1 {{ margin: 0 0 10px; font-size: 22px; letter-spacing: 0; }}
    .subtitle {{ margin: 0; color: var(--muted); font-size: 13px; }}
    main {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 20px 24px 48px;
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 18px;
    }}
    .model-nav {{ position: sticky; top: 12px; align-self: start; }}
    .model-button {{
      width: 100%;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 8px;
      padding: 12px;
      margin: 0 0 10px;
      text-align: left;
      cursor: pointer;
      display: grid;
      gap: 5px;
      font: inherit;
    }}
    .model-button:hover {{ border-color: var(--accent); }}
    .model-button.active {{
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(0, 104, 181, .14);
    }}
    .model-name {{ font-weight: 700; font-size: 15px; }}
    .model-counts {{ color: var(--muted); font-size: 12px; }}
    .model-panel {{ display: none; }}
    .model-panel.active {{ display: block; }}
    .panel-head, .comparison, details {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, .04);
    }}
    .panel-head {{
      padding: 16px;
      display: flex;
      justify-content: space-between;
      gap: 6px 12px;
      align-items: flex-start;
      margin-bottom: 12px;
    }}
    .panel-head h2 {{ margin: 0 0 4px; font-size: 22px; }}
    .report-path {{ color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
    .status-box {{
      min-width: 120px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      text-align: center;
      background: var(--panel-soft);
    }}
    .status-box strong {{ display: block; font-size: 18px; }}
    .status-box span {{ color: var(--muted); font-size: 12px; }}
    .status-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(96px, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }}
    .status-grid div {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      text-align: center;
    }}
    .status-grid strong {{ display: block; font-size: 18px; }}
    .status-grid span {{ color: var(--muted); font-size: 12px; }}
    .model-filter, .comparison-filter {{
      width: 100%;
      max-width: 520px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      margin: 0 0 12px;
      font: inherit;
    }}
    .view-options {{
      display: flex;
      gap: 14px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
      margin: -4px 0 12px;
    }}
    details {{
      margin-bottom: 10px;
      overflow: hidden;
    }}
    summary {{
      cursor: pointer;
      padding: 12px 14px;
      background: var(--panel-soft);
      font-weight: 700;
    }}
    summary span {{ color: var(--muted); font-weight: 400; font-size: 12px; }}
    .table-wrap {{ overflow: auto; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 10px;
      padding: 12px;
    }}
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
      min-width: 0;
    }}
    .metric-card.fail, .metric-card.failed {{
      border-color: rgba(180, 35, 24, .45);
      background: #fffafa;
    }}
    .metric-card.warning {{
      border-color: rgba(181, 71, 8, .45);
      background: #fffdf7;
    }}
    .metric-card.blocked {{
      border-color: rgba(124, 45, 18, .35);
      background: #fffaf4;
    }}
    .hide-not-collected .metric-card.not-collected {{
      display: none;
    }}
    .metric-card-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: flex-start;
      margin-bottom: 8px;
    }}
    .metric-card-head h3 {{
      margin: 0;
      font-size: 14px;
      font-family: var(--mono);
      overflow-wrap: anywhere;
    }}
    .metric-card-meta {{
      display: grid;
      gap: 8px;
      grid-template-columns: 1fr 1fr;
    }}
    .field {{
      min-width: 0;
    }}
    .field.full {{
      grid-column: 1 / -1;
    }}
    .field-label {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .02em;
      margin-bottom: 2px;
    }}
    .field-value {{
      overflow-wrap: anywhere;
      font-size: 13px;
    }}
    .field-value.mono {{
      font-family: var(--mono);
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px;
      vertical-align: top;
      text-align: left;
    }}
    th {{ background: #eef2f7; position: sticky; top: 0; }}
    .metric-name, .metric-path, .metric-value {{ font-family: var(--mono); }}
    .metric-path {{
      color: var(--muted);
      overflow-wrap: anywhere;
    }}
    .metric-value {{ overflow-wrap: anywhere; }}
    .reason {{ color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 2px 7px;
      font-size: 11px;
      font-weight: 700;
    }}
    .pill.pass, .pill.measured {{ color: var(--good); background: #ecfdf3; }}
    .pill.warning {{ color: var(--warn); background: #fffbeb; }}
    .pill.fail, .pill.failed {{ color: var(--bad); background: #fef2f2; }}
    .pill.blocked {{ color: #7c2d12; background: #fff7ed; }}
    .pill.not-collected, .pill.not-applicable, .pill.unknown {{
      color: var(--muted);
      background: #f1f4f8;
    }}
    .comparison {{ margin-top: 18px; padding: 14px; }}
    .comparison h2 {{ margin: 0 0 10px; font-size: 18px; }}
    code {{
      font-family: var(--mono);
      background: var(--panel-soft);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 1px 4px;
    }}
    .hidden {{ display: none; }}
    @media (max-width: 800px) {{
      main {{ grid-template-columns: 1fr; padding-left: 14px; padding-right: 14px; }}
      .model-nav {{ position: static; }}
      .topbar {{ padding-left: 14px; padding-right: 14px; }}
    }}
  </style>
</head>
<body class="hide-not-collected">
  <header>
    <div class="topbar">
      <h1>OVIQS Target Model Quality Dashboard</h1>
      <p class="subtitle">
        Select a model to inspect its metric status, values, references and report paths.
      </p>
    </div>
  </header>
  <main>
    <aside class="model-nav" aria-label="Model selector">
      {model_buttons}
    </aside>
    <div>
      {model_panels}
      <section class="comparison">
        <h2>Critical Metrics Comparison</h2>
        <input
          class="comparison-filter"
          id="comparison-filter"
          placeholder="Filter comparison metrics"
        >
        <div class="table-wrap">
          <table id="critical-comparison">
            <thead>
              <tr><th>Metric</th><th>Section</th>{critical_headers}</tr>
            </thead>
            <tbody>{critical_rows}</tbody>
          </table>
        </div>
      </section>
    </div>
  </main>
  <script>
    const modelButtons = Array.from(document.querySelectorAll(".model-button"));
    const modelPanels = Array.from(document.querySelectorAll(".model-panel"));
    modelButtons.forEach((button) => {{
      button.addEventListener("click", () => {{
        modelButtons.forEach((item) => item.classList.remove("active"));
        modelPanels.forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(`model-panel-${{button.dataset.model}}`).classList.add("active");
      }});
    }});
    document.querySelectorAll(".model-filter").forEach((input) => {{
      input.addEventListener("input", () => {{
        const panel = document.getElementById(input.dataset.panel);
        const query = input.value.toLowerCase();
        panel.querySelectorAll(".metric-card").forEach((card) => {{
          card.classList.toggle("hidden", !card.innerText.toLowerCase().includes(query));
        }});
      }});
    }});
    document.querySelectorAll(".show-not-collected").forEach((input) => {{
      input.addEventListener("change", () => {{
        document.body.classList.toggle("hide-not-collected", !input.checked);
      }});
    }});
    document.getElementById("comparison-filter").addEventListener("input", (event) => {{
      const query = event.target.value.toLowerCase();
      document.querySelectorAll("#critical-comparison tbody tr").forEach((row) => {{
        row.classList.toggle("hidden", !row.innerText.toLowerCase().includes(query));
      }});
    }});
  </script>
</body>
</html>
"""


def _rows_by_section(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["section"]), []).append(row)
    return grouped


def _model_button(label: str, rows: list[dict[str, Any]], *, active: bool) -> str:
    counts = _status_counts(label, rows)
    active_class = " active" if active else ""
    count_line = (
        f"pass {counts['pass']} · fail {counts['fail'] + counts['failed']} · "
        f"blocked {counts['blocked']} · unknown {counts['unknown']}"
    )
    return f"""
      <button class="model-button{active_class}" type="button" data-model="{_html_id(label)}">
        <span class="model-name">{_h(label)}</span>
        <span class="model-counts">{_h(count_line)}</span>
      </button>
    """


def _model_panel(
    label: str,
    rows: list[dict[str, Any]],
    ordered_sections: list[str],
    *,
    report_path: str,
    active: bool,
) -> str:
    counts = _status_counts(label, rows)
    active_class = " active" if active else ""
    status_items = "\n".join(
        f"<div><strong>{counts[key]}</strong><span>{_h(key.replace('_', ' '))}</span></div>"
        for key in (
            "pass",
            "fail",
            "warning",
            "measured",
            "blocked",
            "not_collected",
            "unknown",
            "failed",
        )
    )
    section_tables = "\n".join(
        _model_section_table(label, section, [row for row in rows if row["section"] == section])
        for section in ordered_sections
    )
    panel_id = f"model-panel-{_html_id(label)}"
    return f"""
      <section id="{panel_id}" class="model-panel{active_class}">
        <div class="panel-head">
          <div>
            <h2>{_h(label)}</h2>
            <div class="report-path">Report: <code>{_h(report_path or "n/a")}</code></div>
          </div>
          <div class="status-box">
            <strong>{counts["measured"]}/{len(rows)}</strong>
            <span>measured metrics</span>
          </div>
        </div>
        <div class="status-grid">{status_items}</div>
        <input
          class="model-filter"
          data-panel="{panel_id}"
          placeholder="Filter metrics for {_h(label)}"
        >
        <label class="view-options">
          <input class="show-not-collected" type="checkbox">
          show not_collected catalog gaps
        </label>
        {section_tables}
      </section>
    """


def _model_section_table(label: str, section: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    body = "\n".join(_model_metric_row(label, row) for row in rows)
    description = SECTION_DESCRIPTIONS.get(section, "Metric group from the input reports.")
    title = section.replace("_", " ").title()
    return f"""
      <details open>
        <summary>
          {_h(title)} <span>{len(rows)} metrics · {_h(description)}</span>
        </summary>
        <div class="metric-grid">{body}</div>
      </details>
    """


def _model_metric_row(label: str, row: dict[str, Any]) -> str:
    status = str(row.get(f"{label}_status") or "unknown")
    value = row.get(label) or ""
    path = row.get(f"{label}_path") or ""
    reason = row.get(f"{label}_reason") or ""
    required_inputs = row.get("required_inputs") or []
    required_inputs_text = (
        ", ".join(str(item) for item in required_inputs)
        if isinstance(required_inputs, list)
        else str(required_inputs)
    )
    metric_note = _metric_explanation(row)
    return f"""
      <article class="metric-card {_status_class(status)}" data-status="{_h(status)}">
        <div class="metric-card-head">
          <h3>{_h(row["metric"])}</h3>
          <span class="pill {_status_class(status)}">{_h(status)}</span>
        </div>
        <div class="metric-card-meta">
          <div class="field">
            <div class="field-label">Value</div>
            <div class="field-value mono">{_h(value or "n/a")}</div>
          </div>
          <div class="field">
            <div class="field-label">Path</div>
            <div class="field-value mono">{_h(path or "not found")}</div>
          </div>
          <div class="field full">
            <div class="field-label">Evidence / reason</div>
            <div class="field-value">{_h(reason or "n/a")}</div>
          </div>
          <div class="field full">
            <div class="field-label">Metric explanation</div>
            <div class="field-value">{_h(metric_note)}</div>
          </div>
          <div class="field full">
            <div class="field-label">Required inputs</div>
            <div class="field-value">{_h(required_inputs_text or "n/a")}</div>
          </div>
          <div class="field full">
            <div class="field-label">Reference</div>
            <div class="field-value">{_h(row.get("reference") or "n/a")}</div>
          </div>
        </div>
      </article>
    """


def _critical_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {
        ("likelihood", "perplexity"),
        ("likelihood", "nll"),
        ("inference_equivalence", "mean_kl"),
        ("inference_equivalence", "p95_kl"),
        ("inference_equivalence", "top1_changed_rate"),
        ("generation", "json_validity"),
        ("generation", "repetition_rate"),
        ("long_context", "lost_in_middle_score"),
        ("serving", "batch_mean_kl"),
        ("serving", "batch_p95_kl"),
        ("serving", "batch_top1_changed_rate"),
        ("serving", "kv_mean_kl"),
        ("rag", "context_precision"),
        ("rag", "context_recall"),
        ("rag", "faithfulness"),
        ("agent", "tool_call_validity"),
        ("agent", "recovery_score"),
        ("performance", "forward_latency_ms_mean"),
        ("performance", "generation_latency_ms"),
    }
    selected = [row for row in rows if (row["section"], row["metric"]) in priority]
    return selected or rows[: min(20, len(rows))]


def _critical_metric_row(row: dict[str, Any], labels: list[str]) -> str:
    cells = "\n".join(_critical_metric_value(row, label) for label in labels)
    return f"""
      <tr>
        <td class="metric-name">{_h(row["metric"])}</td>
        <td>{_h(row["section"])}</td>
        {cells}
      </tr>
    """


def _critical_metric_value(row: dict[str, Any], label: str) -> str:
    status = str(row.get(f"{label}_status") or "unknown")
    value = row.get(label) or "n/a"
    reason = row.get(f"{label}_reason") or ""
    return f"""
      <td>
        <span class="pill {_status_class(status)}">{_h(status)}</span><br>
        <span class="metric-value">{_h(value)}</span><br>
        <span class="reason">{_h(reason)}</span>
      </td>
    """


def _status_counts(label: str, rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = (
        "pass",
        "fail",
        "warning",
        "measured",
        "blocked",
        "not_collected",
        "not_applicable",
        "unknown",
        "failed",
    )
    counts: dict[str, int] = dict.fromkeys(keys, 0)
    for row in rows:
        status = str(row.get(f"{label}_status") or "unknown")
        counts[status if status in counts else "unknown"] += 1
    return counts


def _overview_card(label: str, rows: list[dict[str, Any]]) -> str:
    measured = 0
    warning = 0
    unknown = 0
    key_rows: list[dict[str, Any]] = []
    key_metrics = {
        ("likelihood", "perplexity"),
        ("generation", "json_validity"),
        ("serving", "batch_mean_kl"),
        ("serving", "kv_mean_kl"),
        ("performance", "generation_latency_ms"),
    }
    for row in rows:
        status = str(row.get(f"{label}_status", ""))
        if status == "measured":
            measured += 1
        elif status.startswith("unknown"):
            unknown += 1
        else:
            warning += 1
        if (row["section"], row["metric"]) in key_metrics:
            key_rows.append(row)

    key_items = "\n".join(
        f"<dt>{_h(row['metric'])}</dt><dd>{_h(status_value(row, label) or 'n/a')}</dd>"
        for row in key_rows
    )
    return f"""
      <article class="card">
        <h2>{_h(label)}</h2>
        <dl>
          <dt>Measured</dt><dd>{measured}</dd>
          <dt>Warnings</dt><dd>{warning}</dd>
          <dt>Unknown</dt><dd>{unknown}</dd>
          {key_items}
        </dl>
      </article>
    """


def _section_html(section: str, rows: list[dict[str, Any]], labels: list[str]) -> str:
    title = section.replace("_", " ").title()
    description = SECTION_DESCRIPTIONS.get(section, "Metric group from the input reports.")
    metrics = "\n".join(_metric_html(row, labels) for row in rows)
    return f"""
      <section id="{_html_id(section)}">
        <div class="section-head">
          <h2>{_h(title)}</h2>
          <p>{_h(description)}</p>
        </div>
        {metrics}
      </section>
    """


def _metric_html(row: dict[str, Any], labels: list[str]) -> str:
    metric = str(row["metric"])
    metric_id = f"{row['section']}.{metric}"
    explanation = _metric_explanation(row)
    values = "\n".join(_metric_value_html(row, label) for label in labels)
    paths = "\n".join(
        f"<dt>{_h(label)}</dt><dd><code>{_h(row.get(f'{label}_path') or 'not found')}</code></dd>"
        for label in labels
    )
    sources = _sources_html(row)
    oracle = row.get("oracle") or "Use the listed reference and degradation rule."
    return f"""
        <details class="metric">
          <summary>
            <span class="metric-title">
              <strong>{_h(metric_id)}</strong>
              <span>{_h(explanation)}</span>
            </span>
            <span class="values">{values}</span>
          </summary>
          <div class="details-body">
            <div>
              <h3>Reference</h3>
              <p><strong>Primary:</strong> {_h(row.get("reference") or "n/a")}</p>
              <p><strong>Oracle:</strong> {_h(oracle)}</p>
              <p><strong>Degradation:</strong> {_h(row.get("degradation_rule") or "n/a")}</p>
              {sources}
            </div>
            <div>
              <h3>Report Paths</h3>
              <dl>{paths}</dl>
            </div>
          </div>
        </details>
    """


def _metric_value_html(row: dict[str, Any], label: str) -> str:
    value = row.get(label) or "n/a"
    status = str(row.get(f"{label}_status", "unknown"))
    return f"""
              <span class="value">
                <span class="model">{_h(label)}</span>
                <span class="number">{_h(value)}</span>
                <span class="status {_status_class(status)}">{_h(status)}</span>
              </span>
    """


def _sources_html(row: dict[str, Any]) -> str:
    sources = row.get("sources")
    if not isinstance(sources, list) or not sources:
        return ""
    items = []
    for source in sources[:4]:
        if not isinstance(source, dict):
            continue
        name = _h(source.get("name") or source.get("url") or "source")
        url = source.get("url")
        if isinstance(url, str) and url:
            items.append(f'<li><a href="{_h(url)}">{name}</a></li>')
        else:
            items.append(f"<li>{name}</li>")
    if not items:
        return ""
    return "<h3>Sources</h3><ul>" + "".join(items) + "</ul>"


def _metric_explanation(row: dict[str, Any]) -> str:
    section = str(row["section"])
    metric = str(row["metric"])
    full_metric = f"{section}.{metric}"
    path_note = _path_metric_explanation(full_metric, metric)
    if path_note:
        return path_note
    if metric in METRIC_EXPLANATIONS:
        return METRIC_EXPLANATIONS[metric]
    leaf = metric.rsplit(".", 1)[-1]
    if leaf in METRIC_EXPLANATIONS:
        return METRIC_EXPLANATIONS[leaf]
    oracle = row.get("oracle")
    if isinstance(oracle, str) and oracle:
        return oracle
    reference = row.get("reference")
    if isinstance(reference, str) and reference:
        return f"Measured against {reference}."
    return metric.replace("_", " ").capitalize()


def _path_metric_explanation(full_metric: str, metric: str) -> str:
    leaf = metric.rsplit(".", 1)[-1]
    if full_metric.startswith("long_context.position_bucketed_ppl."):
        bucket = metric.split(".")[-2] if "." in metric else "bucket"
        return _bucket_metric_explanation(bucket, leaf)
    if full_metric.startswith("serving.batch_invariance."):
        return _BATCH_INVARIANCE_EXPLANATIONS.get(leaf, "")
    if full_metric.startswith("generation.ngram_repetition."):
        return _NGRAM_REPETITION_EXPLANATIONS.get(leaf, "")
    if full_metric.startswith("generation.json_validity.") and leaf == "json_valid":
        return "Boolean result of parsing generated text as JSON."
    if full_metric.startswith("agent."):
        if leaf in _AGENT_COUNT_EXPLANATIONS:
            return _AGENT_COUNT_EXPLANATIONS[leaf]
        if leaf == "task_completed":
            return "Boolean trace outcome before conversion to task_completion score."
    if full_metric.startswith("rag."):
        return _RAG_COUNT_EXPLANATIONS.get(leaf, "")
    return ""


def _bucket_metric_explanation(bucket: str, leaf: str) -> str:
    templates = {
        "nll": "Mean NLL for relative position bucket {bucket}.",
        "ppl": "Perplexity for relative position bucket {bucket}.",
        "tokens": "Number of scored tokens in relative position bucket {bucket}.",
    }
    template = templates.get(leaf, "")
    return template.format(bucket=bucket) if template else ""


_BATCH_INVARIANCE_EXPLANATIONS = {
    "mean_kl": "Mean KL drift for the same prompt alone versus inside a batch.",
    "p95_kl": "P95 KL drift for the same prompt alone versus inside a batch.",
    "max_kl": "Worst-position KL drift caused by batching.",
    "mean_js": "Mean JS drift for the same prompt alone versus inside a batch.",
    "p95_js": "P95 JS drift for the same prompt alone versus inside a batch.",
    "mean_entropy_drift": "Mean entropy shift introduced by batch execution.",
    "mean_logit_cosine": "Mean logit-vector cosine similarity for single versus batched execution.",
    "top1_changed_rate": "Share of positions whose top-1 token changes under batching.",
    "top10_overlap": "Average top-10 token-set overlap for single versus batched execution.",
}

_NGRAM_REPETITION_EXPLANATIONS = {
    "n": "N-gram order used for repetition scoring.",
    "total_ngrams": "Number of generated n-grams considered by the repetition check.",
}

_AGENT_COUNT_EXPLANATIONS = {
    "checked_steps": "Number of trace steps checked for policy violations.",
    "recovered_tool_errors": "Number of tool errors followed by a recovery action.",
    "redundant_tool_calls": "Number of tool calls detected as redundant.",
    "tool_calls": "Number of tool calls in the trace.",
    "tool_errors": "Number of tool-error steps in the trace.",
    "valid_tool_calls": "Number of tool calls satisfying schema and policy checks.",
}

_RAG_COUNT_EXPLANATIONS = {
    "expected_evidence": "Number of evidence strings expected for the RAG fixture.",
    "matched_evidence": "Number of expected evidence strings found in retrieved context.",
}


def _status_class(status: str) -> str:
    return status.replace("_", "-")


def _html_id(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.lower()).strip("-")


def _h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def status_value(row: dict[str, Any], label: str) -> str:
    value = row.get(label, "")
    status = row.get(f"{label}_status", "")
    reason = row.get(f"{label}_reason", "")
    if not value:
        return f"{status}: {reason}" if reason else str(status)
    return f"{value} ({status})"


def _escape_md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


class ReferenceComparisonAdapter:
    def render(
        self,
        comparison: dict[str, Any],
        format_name: str,
    ) -> str:
        return render_reference_comparison(comparison, format_name)
