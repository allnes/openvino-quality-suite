from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SuiteModelSummary:
    """Headline figures for one model in a multi-report suite bundle.

    These are the few numbers shown on the suite index page; the full detail
    lives in the per-model bundle linked by ``bundle_dir``.
    """

    label: str
    bundle_dir: str
    model: str | None = None
    reference_model: str | None = None
    overall_status: str = "unknown"
    perplexity: float | None = None
    mean_kl: float | None = None
    mean_logit_cosine: float | None = None
    top1_changed_rate: float | None = None
    forward_latency_ms_mean: float | None = None
    tokens_per_second_forward: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SuiteBundle:
    """Result of building a suite bundle: an index linking per-model bundles."""

    root: str
    index_html: str
    comparison_html: str | None
    model_bundles: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_bundles"] = list(self.model_bundles)
        return payload


def _num(payload: Any, key: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def build_suite_model_summary(
    label: str,
    report: dict[str, Any],
    bundle_dir: str,
) -> SuiteModelSummary:
    """Extract the suite-index headline figures from an EvaluationReport dict."""
    run = report.get("run", {}) if isinstance(report.get("run"), dict) else {}
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    likelihood = report.get("likelihood", {})
    equivalence = report.get("inference_equivalence", {})
    performance = report.get("performance", {})
    return SuiteModelSummary(
        label=label,
        bundle_dir=bundle_dir,
        model=run.get("model"),
        reference_model=run.get("reference_model") or run.get("reference"),
        overall_status=str(summary.get("overall_status") or summary.get("overall") or "unknown"),
        perplexity=_num(likelihood, "perplexity"),
        mean_kl=_num(equivalence, "mean_kl"),
        mean_logit_cosine=_num(equivalence, "mean_logit_cosine"),
        top1_changed_rate=_num(equivalence, "top1_changed_rate"),
        forward_latency_ms_mean=_num(performance, "forward_latency_ms_mean"),
        tokens_per_second_forward=_num(performance, "tokens_per_second_forward"),
    )


__all__ = ["SuiteBundle", "SuiteModelSummary", "build_suite_model_summary"]
