"""Canonical evaluation metrics for HNX26EPS04.

Definitions (authoritative):

CER  = Levenshtein(chars) / max(len(reference_chars), 1)
WER  = Levenshtein(words) / max(len(reference_words), 1)
Exact match = 1 if normalize(ref)==normalize(hyp) else 0

FCE (False Confident Error):
  prediction is incorrect AND decision_state == ACCEPTED
  Rate = fce_count / total_lines  (or / accepted when noted)

FVE (False Visible Error):
  prediction is incorrect AND the system displays a non-placeholder
  transcription as active text (ACCEPTED or REVIEW_REQUIRED with visible text).
  ILLEGIBLE / empty / CROSSED_OUT active-channel exclusions apply.

confidence / model_score:
  Unless calibration_is_calibrated==True, numeric scores are UNCALIBRATED
  ranking/consistency scores — NOT P(correct).

Selective risk:
  CER restricted to ACCEPTED lines only (empty accepted → 0.0 by convention,
  with coverage=0 so it cannot look better than full-coverage systems).

Risk-coverage:
  Sort lines by descending confidence among those with a visible hypothesis;
  compute CER/FCE/accepted-accuracy at coverage fractions of the ranked pool.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from rapidfuzz.distance import Levenshtein


PLACEHOLDERS = {"", "[ILLEGIBLE]", "[REVIEW REQUIRED]", "[ILLEGIBLE]."}


def normalize(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate (normalized Levenshtein / |reference|)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return Levenshtein.distance(reference, hypothesis) / max(len(reference), 1)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate."""
    ref_w = reference.split()
    hyp_w = hypothesis.split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return Levenshtein.distance(ref_w, hyp_w) / max(len(ref_w), 1)


def exact_match(reference: str, hypothesis: str) -> float:
    return 1.0 if normalize(reference) == normalize(hypothesis) else 0.0


def line_correct(reference: str, hypothesis: str, *, max_cer: float = 0.15) -> bool:
    if normalize(reference) == normalize(hypothesis):
        return True
    if not reference.strip():
        hyp = hypothesis.strip()
        return (not hyp) or hyp in PLACEHOLDERS
    # Compare on normalized strings for fairness
    return cer(normalize(reference), normalize(hypothesis)) <= max_cer


@dataclass
class LineEval:
    reference: str
    hypothesis: str
    decision_state: str
    confidence: float
    correct: bool
    false_confident_error: bool
    false_visible_error: bool


@dataclass
class AggregateEval:
    total_lines: int = 0
    accepted: int = 0
    review: int = 0
    illegible: int = 0
    crossed_out: int = 0
    correct_accepted: int = 0
    incorrect_accepted: int = 0
    correct_review: int = 0
    incorrect_review: int = 0
    correct_illegible: int = 0
    incorrect_illegible: int = 0
    fce_count: int = 0
    fve_count: int = 0
    cer: float = 0.0
    wer: float = 0.0
    exact_match: float = 0.0
    fce_rate: float = 0.0
    fce_rate_among_accepted: float = 0.0
    fve_rate: float = 0.0
    review_rate: float = 0.0
    abstention_rate: float = 0.0
    coverage_accepted: float = 0.0
    selective_risk: float = 0.0
    accepted_accuracy: float = 0.0
    risk_coverage: list[dict[str, float]] = field(default_factory=list)
    lines: list[LineEval] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    # Explicit semantics
    confidence_is_calibrated: bool = False
    confidence_meaning: str = (
        "uncalibrated_model_consistency_or_ranking_score_not_P_correct"
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_word_list(
    pairs: list[dict[str, Any]],
    *,
    full_reference: Optional[str] = None,
    full_hypothesis: Optional[str] = None,
    max_line_cer: float = 0.15,
    risk_coverage_points: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
) -> AggregateEval:
    """Evaluate aligned line pairs.

    pairs items: reference, hypothesis, decision_state, confidence
    """
    agg = AggregateEval()
    accepted_refs: list[str] = []
    accepted_hyps: list[str] = []

    for p in pairs:
        ref = str(p.get("reference") or "")
        hyp = str(p.get("hypothesis") or "")
        state = str(p.get("decision_state") or "REVIEW_REQUIRED").upper()
        conf = float(p.get("confidence") or 0.0)
        correct = line_correct(ref, hyp, max_cer=max_line_cer)

        fce = (not correct) and state == "ACCEPTED"
        visible = (
            hyp.strip() not in PLACEHOLDERS
            and state not in {"ILLEGIBLE", "CROSSED_OUT"}
        )
        fve = (not correct) and visible and state in {"ACCEPTED", "REVIEW_REQUIRED"}

        agg.lines.append(
            LineEval(
                reference=ref,
                hypothesis=hyp,
                decision_state=state,
                confidence=conf,
                correct=correct,
                false_confident_error=fce,
                false_visible_error=fve,
            )
        )
        agg.total_lines += 1

        if state == "ACCEPTED":
            agg.accepted += 1
            accepted_refs.append(ref)
            accepted_hyps.append(hyp)
            if correct:
                agg.correct_accepted += 1
            else:
                agg.incorrect_accepted += 1
        elif state == "REVIEW_REQUIRED":
            agg.review += 1
            if correct:
                agg.correct_review += 1
            else:
                agg.incorrect_review += 1
        elif state == "ILLEGIBLE":
            agg.illegible += 1
            # Correct illegible only if GT also empty/illegible
            gt_illeg = (not ref.strip()) or normalize(ref) in {
                "[illegible]",
                "illegible",
            }
            if gt_illeg:
                agg.correct_illegible += 1
            else:
                agg.incorrect_illegible += 1
        elif state == "CROSSED_OUT":
            agg.crossed_out += 1

    agg.fce_count = sum(1 for L in agg.lines if L.false_confident_error)
    agg.fve_count = sum(1 for L in agg.lines if L.false_visible_error)

    n = max(agg.total_lines, 1)
    agg.fce_rate = agg.fce_count / n
    agg.fce_rate_among_accepted = (
        agg.incorrect_accepted / agg.accepted if agg.accepted else 0.0
    )
    agg.fve_rate = agg.fve_count / n
    agg.review_rate = agg.review / n
    agg.abstention_rate = (agg.illegible + agg.review) / n
    agg.coverage_accepted = agg.accepted / n
    agg.accepted_accuracy = (
        agg.correct_accepted / agg.accepted if agg.accepted else 0.0
    )

    if full_reference is not None and full_hypothesis is not None:
        agg.cer = cer(full_reference, full_hypothesis)
        agg.wer = wer(full_reference, full_hypothesis)
        agg.exact_match = exact_match(full_reference, full_hypothesis)
    else:
        refs = "\n".join(p.get("reference") or "" for p in pairs)
        hyps = "\n".join(
            (p.get("hypothesis") or "")
            for p in pairs
            if str(p.get("decision_state") or "").upper() != "CROSSED_OUT"
        )
        agg.cer = cer(refs, hyps)
        agg.wer = wer(refs, hyps)
        agg.exact_match = exact_match(refs, hyps)

    if accepted_refs:
        agg.selective_risk = cer("\n".join(accepted_refs), "\n".join(accepted_hyps))
    else:
        agg.selective_risk = 0.0
        agg.extras["selective_risk_note"] = (
            "No accepted lines; selective_risk=0 by convention — "
            "interpret with coverage_accepted=0 (abstention cannot cheat)."
        )

    agg.risk_coverage = risk_coverage_curve(
        agg.lines, points=risk_coverage_points, max_line_cer=max_line_cer
    )
    return agg


def risk_coverage_curve(
    lines: list[LineEval],
    *,
    points: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    max_line_cer: float = 0.15,
) -> list[dict[str, float]]:
    """Risk-coverage over lines ranked by confidence (desc).

    Prevents 'ILLEGIBLE everywhere' from looking superior: at coverage=1.0
    all lines are forced into the evaluated pool.
    """
    if not lines:
        return [
            {
                "coverage": p,
                "cer": 0.0,
                "wer": 0.0,
                "fce": 0.0,
                "accepted_accuracy": 0.0,
                "n": 0,
            }
            for p in points
        ]

    ranked = sorted(lines, key=lambda L: L.confidence, reverse=True)
    n = len(ranked)
    out: list[dict[str, float]] = []
    for p in points:
        k = max(1, int(round(p * n)))
        k = min(k, n)
        subset = ranked[:k]
        refs = "\n".join(L.reference for L in subset)
        hyps = "\n".join(L.hypothesis for L in subset)
        # Treat top-k as "accepted" for this curve point
        fce = sum(
            1
            for L in subset
            if (not line_correct(L.reference, L.hypothesis, max_cer=max_line_cer))
        ) / max(k, 1)
        correct = sum(
            1
            for L in subset
            if line_correct(L.reference, L.hypothesis, max_cer=max_line_cer)
        )
        out.append(
            {
                "coverage": float(p),
                "cer": cer(refs, hyps),
                "wer": wer(refs, hyps),
                "fce": float(fce),
                "accepted_accuracy": correct / max(k, 1),
                "n": float(k),
            }
        )
    return out
