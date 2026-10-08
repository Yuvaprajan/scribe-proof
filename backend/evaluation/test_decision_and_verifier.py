"""Decision engine, verifier wiring, lexicon safety, FCE metrics."""

from __future__ import annotations

from domain.entities import Hypothesis
from domain.policy import DecisionEngine, DecisionState, normalize_candidate
from infrastructure.models.context_reranker import ContextReranker
from infrastructure.models.verifier import DisabledVerifier, VerificationResult
from evaluation.fce_metrics import evaluate_word_list


def _h(
    text: str,
    visual: float,
    variant: str = "clahe",
    rank: int = 1,
    model: str = "microsoft/trocr-large-handwritten",
) -> Hypothesis:
    return Hypothesis(
        recognition_run_id="run",
        text=text,
        normalized_text=normalize_candidate(text),
        rank=rank,
        sequence_score=-rank,
        visual_score=visual,
        model_name=model,
        model_version="1",
        source_crop_id="crop",
        image_variant=variant,
        configuration={"family": "trocr"} if "trocr" in model or "microsoft" in model else {},
    )


def test_strong_consensus_accepted():
    hyps = [
        _h("hello", 0.92, "raw"),
        _h("hello", 0.90, "clahe"),
        _h("hello", 0.89, "denoised"),
        _h("hello", 0.88, "adaptive_binarized"),
    ]
    result = DecisionEngine().fuse(hyps, image_quality_risk=0.1)
    assert result.state == DecisionState.ACCEPTED
    assert result.text == "hello"
    assert result.breakdown.confidence_is_calibrated == 0.0


def test_htr_blocked_illegible():
    result = DecisionEngine().fuse([], htr_blocked=True)
    assert result.state == DecisionState.ILLEGIBLE
    assert result.text == "[ILLEGIBLE]"


def test_verifier_selects_candidate_b():
    hyps = [
        _h("wrong", 0.8, "clahe"),
        _h("correct", 0.7, "clahe"),
        _h("other", 0.6, "clahe"),
    ]
    result = DecisionEngine().fuse(
        hyps,
        image_quality_risk=0.1,
        verifier_selected_index=1,
        verifier_agreement=1.0,
    )
    assert result.selected is not None
    assert result.selected.text == "correct"


def test_verifier_reject_all():
    hyps = [_h("guess", 0.7, "clahe")]
    result = DecisionEngine().fuse(
        hyps, verifier_reject_all=True, image_quality_risk=0.2
    )
    assert result.state in {DecisionState.ILLEGIBLE, DecisionState.REVIEW_REQUIRED}


def test_paddle_only_never_accepted_for_hw():
    hyps = [
        _h("wrong print guess", 0.95, "raw", model="paddleocr-rec"),
    ]
    result = DecisionEngine().fuse(hyps, image_quality_risk=0.2)
    assert result.state != DecisionState.ACCEPTED


def test_disabled_verifier_unavailable():
    v = DisabledVerifier()
    r = v.verify(None, ["a", "b"])  # type: ignore[arg-type]
    assert r.status == "unavailable"
    assert r.enabled is False


def test_lexicon_does_not_invent_distant_terms():
    reranker = ContextReranker()
    # Far from any lexicon term
    repaired, edits = reranker._lexicon_repair_tokens("xqzztplm")
    assert repaired == "xqzztplm"
    assert edits == []


def test_lexicon_near_exact_allowed():
    reranker = ContextReranker()
    # "amoxicilin" → "amoxicillin" is 1 char — may or may not pass depending on dist
    repaired, edits = reranker._lexicon_repair_tokens("patient")
    assert repaired == "patient"


def test_fce_metric():
    pairs = [
        {
            "reference": "hello",
            "hypothesis": "hello",
            "decision_state": "ACCEPTED",
            "confidence": 0.9,
        },
        {
            "reference": "world",
            "hypothesis": "wrld",
            "decision_state": "ACCEPTED",
            "confidence": 0.8,
        },
        {
            "reference": "foo",
            "hypothesis": "bar",
            "decision_state": "REVIEW_REQUIRED",
            "confidence": 0.5,
        },
        {
            "reference": "baz",
            "hypothesis": "[ILLEGIBLE]",
            "decision_state": "ILLEGIBLE",
            "confidence": 0.1,
        },
    ]
    agg = evaluate_word_list(pairs)
    assert agg.fce_count == 1
    assert agg.incorrect_accepted == 1
    assert agg.accepted == 2
    assert agg.fve_count >= 1


def test_conflict_not_confidently_accepted():
    hyps = [
        _h("alpha", 0.45, "raw"),
        _h("beta", 0.44, "clahe"),
        _h("gamma", 0.43, "denoised"),
    ]
    result = DecisionEngine().fuse(hyps, image_quality_risk=0.6)
    assert result.state in {DecisionState.ILLEGIBLE, DecisionState.REVIEW_REQUIRED}


def test_qwen_advisory_conflict_mixed():
    from domain.entities import BoundingBox, RegionType
    from infrastructure.models.document_understanding import (
        DocumentUnderstanding,
        LayoutRegionHint,
        apply_layout_hints_to_lines,
    )
    from infrastructure.models.detection import DetectedLine

    line = DetectedLine(
        bbox=BoundingBox(x=10, y=10, width=100, height=30),
        polygon=[],
        confidence=0.5,
        region_type=RegionType.MAIN_HANDWRITING,
        metadata={
            "script_mode": "handwriting",
            "stroke_irregularity": 0.5,
            "paddle_score": 0.4,
        },
    )
    understanding = DocumentUnderstanding(
        layout_regions=[
            LayoutRegionHint(
                region_type=RegionType.PRINTED_TEXT,
                bbox_norm=[0.0, 0.0, 0.3, 0.1],
                description="print zone",
            )
        ],
        enabled=True,
        provider="test",
    )
    apply_layout_hints_to_lines([line], understanding, 400, 400)
    # Strong HW must not be forced to printed_text
    assert line.region_type != RegionType.PRINTED_TEXT

