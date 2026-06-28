# ruff: noqa: E501

from __future__ import annotations

from html import escape
from typing import Any

from oviqs.domain.reporting import SuiteModelSummary

_STATUS_CLASS = {"pass": "pass", "warning": "warn", "fail": "fail"}


class SuiteIndexRenderer:
    """Render the landing page for a multi-model suite bundle.

    Produces a self-contained index.html with a fidelity-first summary table that
    links to each per-model dashboard bundle and to the cross-model comparison.
    """

    format_name = "suite-index"

    def render(
        self,
        summaries: list[SuiteModelSummary],
        *,
        comparison_href: str | None = None,
        title: str = "OVIQS — OpenVINO Inference Quality Suite",
    ) -> str:
        rows = "\n".join(_row(s) for s in summaries)
        cta = (
            f'<a class="cta" href="{escape(comparison_href)}">Open cross-model comparison &rarr;</a>'
            if comparison_href
            else ""
        )
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
 body {{ margin:0; font:14px/1.55 system-ui,-apple-system,Segoe UI,sans-serif; background:#f3f5f8; color:#17202a; }}
 header {{ padding:26px 34px 18px; background:linear-gradient(180deg,#fff,#f7f9fc); border-bottom:1px solid #d8dee6; }}
 main {{ padding:22px 34px 48px; max-width:1100px; }}
 h1 {{ margin:0 0 6px; font-size:25px; }}
 h2 {{ font-size:18px; border-left:3px solid #1f4e96; padding-left:9px; margin:30px 0 12px; }}
 .lede {{ color:#5d6673; max-width:800px; margin:0 0 8px; }}
 table {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid #d8dee6; border-radius:8px; overflow:hidden; }}
 th,td {{ padding:9px 11px; border-bottom:1px solid #eef2f6; text-align:left; vertical-align:top; }}
 th {{ background:#eef2f6; color:#5d6673; font-size:12px; text-transform:uppercase; letter-spacing:.02em; }}
 tr:hover td {{ background:#f7fafe; }}
 a {{ color:#1f4e96; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
 .art {{ color:#5d6673; font-size:11.5px; font-family:ui-monospace,Menlo,monospace; }}
 .kl {{ font-weight:650; }}
 .pill {{ display:inline-block; padding:1px 8px; border-radius:999px; font-size:12px; font-weight:650; border:1px solid currentColor; }}
 .pass {{ color:#147a3f; }} .warn {{ color:#946200; }} .fail {{ color:#b42318; }} .unk {{ color:#5d6673; }}
 .cta {{ display:inline-block; margin:6px 10px 0 0; padding:9px 14px; border:1px solid #1f4e96; border-radius:7px; color:#1f4e96; font-weight:650; }}
 .note {{ color:#5d6673; font-size:12.5px; }}
</style></head>
<body>
<header>
 <h1>{escape(title)}</h1>
 <p class="lede">Export-fidelity scorecard across {len(summaries)} models. Inference equivalence is measured as <strong>PyTorch reference vs OpenVINO export</strong> — how closely each export tracks the source framework. Lower <code>mean&nbsp;KL</code> = closer; <code>logit&nbsp;cosine</code> &rarr;1 is ideal.</p>
 {cta}
</header>
<main>
 <h2>Models at a glance</h2>
 <p class="lede">Click a model for its full dashboard (fidelity panel, surface glossary, all metric tables).</p>
 <table>
  <thead><tr><th>Model</th><th>Overall</th><th>PPL</th><th>mean&nbsp;KL</th><th>logit&nbsp;cos</th><th>top1&nbsp;&Delta;</th><th>fwd&nbsp;ms</th><th>tok/s</th><th>Artifacts</th></tr></thead>
  <tbody>
{rows}
  </tbody>
 </table>
 <p class="note">Status reflects each model's own gates. A non-zero export drift is expected for quantized models and is reported as <em>measured</em>, not failed, unless a gate is configured.</p>
</main>
</body></html>
"""


def _row(s: SuiteModelSummary) -> str:
    cls = _STATUS_CLASS.get(s.overall_status, "unk")
    art = escape(s.model or "")
    return (
        f'<tr><td><a href="{escape(s.bundle_dir)}/dashboard.html">{escape(s.label)}</a>'
        f'<div class="art">{art}</div></td>'
        f'<td><span class="pill {cls}">{escape(s.overall_status.upper())}</span></td>'
        f"<td>{_fmt(s.perplexity)}</td>"
        f'<td class="kl">{_fmt(s.mean_kl)}</td>'
        f"<td>{_fmt(s.mean_logit_cosine)}</td>"
        f"<td>{_fmt(s.top1_changed_rate)}</td>"
        f"<td>{_fmt(s.forward_latency_ms_mean)}</td>"
        f"<td>{_fmt(s.tokens_per_second_forward)}</td>"
        f'<td><a href="{escape(s.bundle_dir)}/dashboard.html">dashboard</a> · '
        f'<a href="{escape(s.bundle_dir)}/index.md">md</a> · '
        f'<a href="{escape(s.bundle_dir)}/metrics.csv">csv</a></td></tr>'
    )


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return "—"
    return escape(str(value))


__all__ = ["SuiteIndexRenderer"]
