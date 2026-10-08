"""Deterministic evidence decision engine — never invent; prefer handwriting HTR."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz.distance import Levenshtein

from .entities import DecisionState, Hypothesis


HIGH_RISK_PATTERNS = [
    re.compile(r"\d"),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(r"[A-Z][a-z]+"),
    re.compile(r"[A-Z0-9]{4,}"),
]


@dataclass
class DecisionThresholds:
    accepted_min_score: float = 0.72
    accepted_max_uncertainty: float = 0.28
    review_max_uncertainty: float = 0.62
    illegible_min_visual: float = 0.28
    consensus_min_support: int = 1
    high_risk_accepted_min_score: float = 0.82
    high_risk_max_uncertainty: float = 0.22
    crossed_out_threshold: float = 0.55
    require_trocr_for_accepted: bool = True
    paddle_only_max_state: str = "REVIEW_REQUIRED"
    # Conservative abstention
    max_ensemble_disagreement_accept: float = 0.40
    max_cross_model_accept: float = 0.35
    min_variant_agreement_accept: float = 0.5

    def as_numeric_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, bool):
                out[k] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                out[k] = float(v)
        return out


@dataclass
class ScoreBreakdown:
    """Separated score semantics — do NOT treat model_score as P(correct)."""

    model_score: float = 0.0  # raw OCR visual / beam consistency
    visual_evidence_score: float = 0.0
    agreement_score: float = 0.0
    uncertainty_score: float = 0.0
    # Legacy aliases used by UI / existing clients
    normalized_visual_score: float = 0.0
    variant_agreement: float = 0.0
    character_stability: float = 0.0
    context_score: float = 0.0
    verifier_agreement: float = 0.0
    decoder_entropy: float = 0.0
    ensemble_disagreement: float = 0.0
    image_quality_risk: float = 0.0
    verifier_disagreement: float = 0.0
    final_score: float = 0.0
    uncertainty: float = 0.0
    # Explicitly NOT statistically calibrated until labeled calibration exists
    calibrated_confidence: Optional[float] = None
    confidence_is_calibrated: float = 0.0  # 0 = uncalibrated display score


@dataclass
class FusionResult:
    selected: Optional[Hypothesis]
    text: str
    state: DecisionState
    confidence: float
    uncertainty: float
    final_score: float
    alternatives: list[str]
    reason: str
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    is_high_risk: bool = False


def normalize_candidate(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def is_high_risk_entity(text: str) -> bool:
    return any(p.search(text) for p in HIGH_RISK_PATTERNS)


def is_trocr_hyp(h: Hypothesis) -> bool:
    name = (h.model_name or "").lower()
    family = (h.configuration or {}).get("family", "")
    return "trocr" in name or "microsoft" in name or family == "trocr"


def is_paddle_hyp(h: Hypothesis) -> bool:
    return "paddle" in (h.model_name or "").lower()


def _softmax_entropy(scores: list[float]) -> float:
    if not scores:
        return 1.0
    shifted = [s - max(scores) for s in scores]
    exps = [math.exp(s) for s in shifted]
    total = sum(exps) or 1.0
    probs = [e / total for e in exps]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    max_ent = math.log(len(probs)) if len(probs) > 1 else 1.0
    return min(1.0, entropy / max_ent)


_DOMAIN_HINTS = {
    "patient",
    "allergy",
    "nkda",
    "amoxicillin",
    "rx",
    "mg",
    "notes",
    "date",
    "follow",
    "fever",
    "tid",
    "bid",
    "dose",
}


def _domain_hint_score(text: str) -> float:
    toks = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    if not toks:
        return 0.0
    return min(1.0, 0.25 * len(toks & _DOMAIN_HINTS))


def _cluster_hypotheses(hyps: list[Hypothesis]) -> list[list[Hypothesis]]:
    clusters: list[list[Hypothesis]] = []
    for h in hyps:
        placed = False
        for cluster in clusters:
            rep = cluster[0].normalized_text
            dist = Levenshtein.normalized_distance(h.normalized_text, rep)
            if dist <= 0.28:
                cluster.append(h)
                placed = True
                break
        if not placed:
            clusters.append([h])

    def cluster_key(c: list[Hypothesis]) -> tuple:
        trocr_n = sum(1 for h in c if is_trocr_hyp(h))
        best = max(h.visual_score for h in c)
        domain = max((_domain_hint_score(h.text) for h in c), default=0.0)
        has_safe_edit = any((h.configuration or {}).get("safe_edit") for h in c)
        return (-trocr_n, -domain, -int(has_safe_edit), -best, -len(c))

    clusters.sort(key=cluster_key)
    return clusters


def _character_stability(cluster: list[Hypothesis]) -> float:
    texts = [h.normalized_text for h in cluster if h.normalized_text]
    if len(texts) <= 1:
        return 1.0 if texts else 0.0
    total = 0.0
    pairs = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            total += 1.0 - Levenshtein.normalized_distance(texts[i], texts[j])
            pairs += 1
    return total / pairs if pairs else 0.0


def _pick_best(cluster: list[Hypothesis]) -> Hypothesis:
    trocr = [h for h in cluster if is_trocr_hyp(h)]
    pool = trocr or cluster
    return max(
        pool,
        key=lambda h: (
            h.visual_score
            + (0.03 if (h.configuration or {}).get("safe_edit") else 0.0)
            + 0.04 * _domain_hint_score(h.text)
        ),
    )


class DecisionEngine:
    def __init__(self, thresholds: Optional[DecisionThresholds] = None):
        self.thresholds = thresholds or DecisionThresholds()

    def fuse(
        self,
        hypotheses: list[Hypothesis],
        *,
        is_crossed_out: bool = False,
        image_quality_risk: float = 0.3,
        context_score: float = 0.0,
        verifier_agreement: float = 0.0,
        verifier_disagreement: float = 0.0,
        script_mode: str = "handwriting",
        region_type: str = "",
        verifier_selected_index: Optional[int] = None,
        verifier_reject_all: bool = False,
        htr_blocked: bool = False,
        routing_uncertain: bool = False,
        strike_uncertain: bool = False,
    ) -> FusionResult:
        print_mode = script_mode == "print" or region_type in {
            "printed_text",
            "table",
        }
        mixed_mode = script_mode == "mixed" or routing_uncertain

        if htr_blocked and not print_mode:
            return FusionResult(
                selected=None,
                text="[ILLEGIBLE]",
                state=DecisionState.ILLEGIBLE,
                confidence=0.0,
                uncertainty=1.0,
                final_score=0.0,
                alternatives=[h.text for h in hypotheses[:5] if h.text],
                reason=(
                    "HTR blocked: invalid/page-sized/merged crop — "
                    "refusing to run line OCR on non-line geometry."
                ),
                is_high_risk=False,
            )

        if verifier_reject_all:
            alts = [h.text for h in hypotheses[:5] if h.text]
            return FusionResult(
                selected=None,
                text="[ILLEGIBLE]" if not alts else (alts[0]),
                state=DecisionState.ILLEGIBLE if not alts else DecisionState.REVIEW_REQUIRED,
                confidence=0.15,
                uncertainty=0.9,
                final_score=0.15,
                alternatives=alts,
                reason="Verifier rejected all candidates as visually unsupported.",
                is_high_risk=bool(alts and is_high_risk_entity(alts[0])),
            )

        # Apply verifier selection among existing hypotheses (candidate-bound)
        if verifier_selected_index is not None and hypotheses:
            idx = int(verifier_selected_index)
            if 0 <= idx < len(hypotheses) and hypotheses[idx].text:
                chosen = hypotheses[idx]
                # Move chosen to front of ranking for fusion
                hypotheses = [chosen] + [h for i, h in enumerate(hypotheses) if i != idx]
                verifier_agreement = max(verifier_agreement, 1.0)

        if is_crossed_out:
            best = max(hypotheses, key=lambda h: h.visual_score) if hypotheses else None
            text = best.text if best else ""
            return FusionResult(
                selected=best,
                text=text,
                state=DecisionState.CROSSED_OUT,
                confidence=best.visual_score if best else 0.0,
                uncertainty=0.5,
                final_score=best.visual_score if best else 0.0,
                alternatives=[h.text for h in hypotheses[:5] if h.text],
                reason="Visual cancellation detected; retained separately from active text.",
                is_high_risk=is_high_risk_entity(text) if text else False,
            )

        if strike_uncertain and hypotheses:
            # Preserve OCR evidence; force review rather than crossed-out deletion
            pass  # handled after scoring

        valid = [h for h in hypotheses if h.normalized_text]
        if not valid:
            return FusionResult(
                selected=None,
                text="[ILLEGIBLE]",
                state=DecisionState.ILLEGIBLE,
                confidence=0.0,
                uncertainty=1.0,
                final_score=0.0,
                alternatives=[],
                reason="No OCR candidates produced readable text.",
            )

        trocr_valid = [h for h in valid if is_trocr_hyp(h)]
        paddle_valid = [h for h in valid if is_paddle_hyp(h)]

        clusters = _cluster_hypotheses(valid)
        top = clusters[0]

        if print_mode and not mixed_mode:
            for c in clusters:
                if any(is_paddle_hyp(h) for h in c):
                    top = c
                    break
            best = max(top, key=lambda h: h.visual_score)
            if paddle_valid and not is_paddle_hyp(best):
                best = max(paddle_valid, key=lambda h: h.visual_score)
        else:
            best = _pick_best(top)
            if not any(is_trocr_hyp(h) for h in top) and trocr_valid:
                for c in clusters:
                    if any(is_trocr_hyp(h) for h in c):
                        top = c
                        best = _pick_best(top)
                        break

        # If verifier selected a specific hyp, prefer it as best when still valid
        if verifier_selected_index is not None and hypotheses:
            sel = hypotheses[0]
            if sel.text:
                best = sel
                # Rebuild top cluster around selected
                top = [h for h in valid if Levenshtein.normalized_distance(
                    h.normalized_text, sel.normalized_text
                ) <= 0.28] or [sel]

        trocr_in_top = [h for h in top if is_trocr_hyp(h)]
        paddle_in_top = [h for h in top if is_paddle_hyp(h)]
        support_pool = (
            paddle_in_top
            if print_mode and not mixed_mode and paddle_in_top
            else (trocr_in_top or top)
        )
        variants = {h.image_variant for h in support_pool} or {
            h.image_variant for h in top
        }
        all_variants = {
            h.image_variant for h in (paddle_valid if print_mode else trocr_valid)
        } or {h.image_variant for h in valid}
        variant_agreement = len(variants) / max(len(all_variants), 1)
        char_stability = _character_stability(support_pool)
        visual = max((h.visual_score for h in support_pool), default=0.0)
        visual = max(0.0, min(1.0, visual))

        cross_model_disagreement = 0.0
        if trocr_valid and paddle_valid:
            t_best = max(trocr_valid, key=lambda h: h.visual_score)
            p_best = max(paddle_valid, key=lambda h: h.visual_score)
            if t_best.normalized_text and p_best.normalized_text:
                cross_model_disagreement = Levenshtein.normalized_distance(
                    t_best.normalized_text, p_best.normalized_text
                )

        cluster_scores = [
            max((h.visual_score for h in c if is_trocr_hyp(h)), default=0.0)
            or max(h.visual_score for h in c)
            for c in clusters
        ]
        if len(clusters) == 1:
            decoder_entropy = max(0.0, 0.15 * (1.0 - visual))
        else:
            decoder_entropy = _softmax_entropy(cluster_scores)

        ensemble_disagreement = 0.0
        if len(clusters) > 1:
            ensemble_disagreement = max(
                1.0 - variant_agreement,
                0.35 * (1.0 - char_stability),
                0.4 * cross_model_disagreement,
            )
        else:
            ensemble_disagreement = max(
                0.12 * (1 - char_stability), 0.35 * cross_model_disagreement
            )

        agreement = 0.5 * variant_agreement + 0.5 * char_stability
        breakdown = ScoreBreakdown(
            model_score=visual,
            visual_evidence_score=visual,
            agreement_score=agreement,
            normalized_visual_score=visual,
            variant_agreement=variant_agreement,
            character_stability=char_stability,
            context_score=max(0.0, min(1.0, context_score)),
            verifier_agreement=max(0.0, min(1.0, verifier_agreement)),
            decoder_entropy=decoder_entropy,
            ensemble_disagreement=min(1.0, ensemble_disagreement),
            image_quality_risk=max(0.0, min(1.0, image_quality_risk)),
            verifier_disagreement=max(0.0, min(1.0, verifier_disagreement)),
            calibrated_confidence=None,
            confidence_is_calibrated=0.0,
        )

        breakdown.final_score = (
            0.45 * breakdown.model_score
            + 0.20 * breakdown.variant_agreement
            + 0.15 * breakdown.character_stability
            + 0.10 * breakdown.context_score
            + 0.10 * breakdown.verifier_agreement
        )
        breakdown.uncertainty = (
            0.35 * breakdown.decoder_entropy
            + 0.35 * breakdown.ensemble_disagreement
            + 0.20 * breakdown.image_quality_risk
            + 0.10 * breakdown.verifier_disagreement
        )
        breakdown.uncertainty_score = breakdown.uncertainty
        # Display confidence = uncalibrated evidence score (explicitly marked)
        display_confidence = breakdown.final_score

        text = best.text
        high_risk = is_high_risk_entity(text)
        alts: list[str] = []
        seen = set()
        for c in clusters:
            t = _pick_best(c).text
            if t and t not in seen:
                alts.append(t)
                seen.add(t)
            if len(alts) >= 5:
                break

        th = self.thresholds
        min_score = (
            th.high_risk_accepted_min_score if high_risk else th.accepted_min_score
        )
        max_unc = (
            th.high_risk_max_uncertainty if high_risk else th.accepted_max_uncertainty
        )

        paddle_only = bool(valid) and not trocr_valid
        trocr_empty = not trocr_valid or all(not h.text for h in trocr_valid)

        # Strong abstention triggers
        abstain = False
        abstain_reason = ""
        if image_quality_risk >= 0.8:
            abstain, abstain_reason = True, "Poor image quality"
        if breakdown.ensemble_disagreement > 0.65 and breakdown.final_score < 0.55:
            abstain, abstain_reason = True, "Heavy candidate disagreement"
        if decoder_entropy > 0.85 and visual < 0.55:
            abstain, abstain_reason = True, "High decoder entropy"
        if mixed_mode and cross_model_disagreement > 0.4:
            abstain, abstain_reason = True, "Uncertain print/handwriting routing"
        if verifier_disagreement >= 1.0:
            abstain, abstain_reason = True, "Verifier disagreement"

        if not print_mode and (paddle_only or (trocr_empty and paddle_valid)):
            if th.require_trocr_for_accepted or True:
                return FusionResult(
                    selected=None if trocr_empty else best,
                    text="[ILLEGIBLE]"
                    if trocr_empty
                    else (best.text if best else "[ILLEGIBLE]"),
                    state=DecisionState.ILLEGIBLE
                    if trocr_empty
                    else DecisionState.REVIEW_REQUIRED,
                    confidence=min(display_confidence, 0.35),
                    uncertainty=max(breakdown.uncertainty, 0.75),
                    final_score=breakdown.final_score,
                    alternatives=alts,
                    reason=(
                        "Print OCR / weak evidence only — refusing confident acceptance. "
                        "Flagged for review or illegible (no reliable handwriting HTR)."
                    ),
                    breakdown=breakdown,
                    is_high_risk=high_risk,
                )

        if abstain and visual < th.illegible_min_visual:
            return FusionResult(
                selected=None,
                text="[ILLEGIBLE]",
                state=DecisionState.ILLEGIBLE,
                confidence=0.1,
                uncertainty=0.95,
                final_score=breakdown.final_score,
                alternatives=alts,
                reason=f"Abstain: {abstain_reason}; insufficient visual evidence.",
                breakdown=breakdown,
                is_high_risk=high_risk,
            )

        if visual < th.illegible_min_visual or (
            len(clusters) > 1
            and breakdown.ensemble_disagreement > 0.7
            and breakdown.final_score < 0.5
        ):
            state = DecisionState.ILLEGIBLE
            out_text = "[ILLEGIBLE]"
            reason = (
                "Insufficient evidence or heavy candidate conflict; "
                "refusing to invent text."
            )
        elif (
            print_mode
            and not mixed_mode
            and is_paddle_hyp(best)
            and visual >= max(0.60, min_score - 0.08)
            and not abstain
        ):
            if breakdown.uncertainty <= max_unc + 0.08:
                state = DecisionState.ACCEPTED
                out_text = text
                reason = (
                    f"Printed-text Paddle OCR accepted "
                    f"(evidence_score={breakdown.final_score:.2f}, "
                    f"unc={breakdown.uncertainty:.2f}; uncalibrated)."
                )
            else:
                state = DecisionState.REVIEW_REQUIRED
                out_text = text
                reason = "Printed text uncertain — flagged for review."
        elif (
            not print_mode
            and not abstain
            and breakdown.final_score >= min_score
            and breakdown.uncertainty <= max_unc
            and len(trocr_in_top) >= max(1, th.consensus_min_support)
            and is_trocr_hyp(best)
            and breakdown.ensemble_disagreement <= th.max_ensemble_disagreement_accept
            and cross_model_disagreement <= th.max_cross_model_accept
            and variant_agreement >= th.min_variant_agreement_accept
        ):
            state = DecisionState.ACCEPTED
            out_text = text
            reason = (
                f"TrOCR consensus (n={len(trocr_in_top)}, variants={len(variants)}); "
                f"evidence_score={breakdown.final_score:.2f} (uncalibrated); "
                f"uncertainty={breakdown.uncertainty:.2f}."
            )
        elif abstain or (
            breakdown.uncertainty <= th.review_max_uncertainty and text
        ):
            state = DecisionState.REVIEW_REQUIRED
            out_text = text
            reason = (
                (
                    f"Abstain→review: {abstain_reason}. "
                    if abstain
                    else ""
                )
                + f"{'Mixed/uncertain routing' if mixed_mode else ('Printed text' if print_mode else 'Handwriting')} "
                f"uncertain — flagged for review "
                f"(evidence_score={breakdown.final_score:.2f}, "
                f"uncertainty={breakdown.uncertainty:.2f}"
                f"{', cross-model disagreement' if cross_model_disagreement > 0.35 else ''}). "
                f"Displayed text is a candidate, not a calibrated acceptance."
            )
        else:
            state = DecisionState.ILLEGIBLE
            out_text = "[ILLEGIBLE]"
            reason = (
                f"High uncertainty ({breakdown.uncertainty:.2f}); marked illegible "
                "instead of guessing."
            )

        if strike_uncertain and state == DecisionState.ACCEPTED:
            state = DecisionState.REVIEW_REQUIRED
            out_text = text
            reason = "Possible strike-through (uncertain) — review; OCR evidence preserved."

        if (
            high_risk
            and state == DecisionState.ACCEPTED
            and breakdown.uncertainty > th.high_risk_max_uncertainty
        ):
            state = DecisionState.REVIEW_REQUIRED
            out_text = text
            reason = "High-risk entity (number/date/name/id) flagged for review."

        if (
            not print_mode
            and cross_model_disagreement > 0.45
            and state == DecisionState.ACCEPTED
        ):
            state = DecisionState.REVIEW_REQUIRED
            out_text = text
            reason = (
                "TrOCR and print-OCR disagree substantially; "
                "downgraded to REVIEW_REQUIRED."
            )

        if state == DecisionState.ACCEPTED and (
            "?" in text
            or text.count(".") > 3
            or re.search(
                r"\b[a-z]{1,2}\b.*\b[a-z]{1,2}\b.*\b[a-z]{1,2}\b", text.lower()
            )
            is not None
            and len(text) < 12
        ):
            state = DecisionState.REVIEW_REQUIRED
            out_text = text
            reason = "Decode artifacts present; flagged for review instead of accepting."

        # Cap displayed confidence for non-accepted states so UI does not imply trust
        if state == DecisionState.REVIEW_REQUIRED:
            display_confidence = min(display_confidence, 0.55)
        elif state == DecisionState.ILLEGIBLE:
            display_confidence = min(display_confidence, 0.2)
            out_text = "[ILLEGIBLE]"
            best = None

        return FusionResult(
            selected=best if state != DecisionState.ILLEGIBLE else None,
            text=out_text,
            state=state,
            confidence=display_confidence,
            uncertainty=breakdown.uncertainty,
            final_score=breakdown.final_score,
            alternatives=alts,
            reason=reason,
            breakdown=breakdown,
            is_high_risk=high_risk,
        )
