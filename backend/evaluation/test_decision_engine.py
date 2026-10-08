"""Quick offline checks for the deterministic decision engine."""

from __future__ import annotations

from domain.entities import Hypothesis
from domain.policy import DecisionEngine, DecisionState, normalize_candidate


def _h(
    text: str,
    visual: float,
    variant: str,
    rank: int = 1,
    model: str = "microsoft/trocr-base-handwritten",
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
    )


def test_strong_consensus_accepted():
    hyps = [
        _h("hello", 0.9, "raw"),
        _h("hello", 0.88, "clahe"),
        _h("hello", 0.87, "denoised"),
        _h("hello", 0.86, "adaptive_binarized"),
    ]
    result = DecisionEngine().fuse(hyps, image_quality_risk=0.1)
    assert result.state == DecisionState.ACCEPTED
    assert result.text == "hello"


def test_conflict_illegible():
    hyps = [
        _h("alpha", 0.4, "raw"),
        _h("beta", 0.39, "clahe"),
        _h("gamma", 0.38, "denoised"),
        _h("delta", 0.37, "adaptive_binarized"),
    ]
    result = DecisionEngine().fuse(hyps, image_quality_risk=0.6)
    assert result.state in {DecisionState.ILLEGIBLE, DecisionState.REVIEW_REQUIRED}


def test_paddle_only_never_accepted():
    hyps = [
        _h("wrong print guess", 0.95, "raw", model="paddleocr-rec"),
        _h("other guess", 0.9, "clahe", model="paddleocr-rec"),
    ]
    result = DecisionEngine().fuse(hyps, image_quality_risk=0.2)
    assert result.state in {DecisionState.ILLEGIBLE, DecisionState.REVIEW_REQUIRED}
    assert result.state != DecisionState.ACCEPTED


def test_print_mode_paddle_accepted():
    hyps = [
        _h("Patient Name: Jordan Miles", 0.92, "clahe", model="paddleocr-rec"),
    ]
    result = DecisionEngine().fuse(
        hyps, image_quality_risk=0.1, script_mode="print", region_type="printed_text"
    )
    assert result.state == DecisionState.ACCEPTED
    assert "Jordan" in result.text


def test_cross_model_disagreement_downgrades():
    hyps = [
        _h("amoxicillin", 0.9, "clahe"),
        _h("amoxicillin", 0.88, "denoised"),
        _h("completely different", 0.9, "raw", model="paddleocr-rec"),
    ]
    result = DecisionEngine().fuse(hyps, image_quality_risk=0.1)
    assert result.state == DecisionState.REVIEW_REQUIRED
    assert "amoxicillin" in result.text.lower()


def test_crossed_out():
    hyps = [_h("old dose", 0.7, "raw")]
    result = DecisionEngine().fuse(hyps, is_crossed_out=True)
    assert result.state == DecisionState.CROSSED_OUT


def test_empty_illegible():
    result = DecisionEngine().fuse([])
    assert result.state == DecisionState.ILLEGIBLE
    assert result.text == "[ILLEGIBLE]"


if __name__ == "__main__":
    test_strong_consensus_accepted()
    test_conflict_illegible()
    test_paddle_only_never_accepted()
    test_print_mode_paddle_accepted()
    test_cross_model_disagreement_downgrades()
    test_crossed_out()
    test_empty_illegible()
    print("decision engine checks passed")
