"""Backward-compatible re-export of canonical metrics (see metrics_core.py)."""

from __future__ import annotations

from evaluation.metrics_core import (  # noqa: F401
    PLACEHOLDERS,
    AggregateEval,
    LineEval,
    cer,
    exact_match,
    evaluate_word_list,
    line_correct,
    normalize,
    risk_coverage_curve,
    wer,
)

__all__ = [
    "PLACEHOLDERS",
    "AggregateEval",
    "LineEval",
    "cer",
    "exact_match",
    "evaluate_word_list",
    "line_correct",
    "normalize",
    "risk_coverage_curve",
    "wer",
]
