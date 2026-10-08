"""Extreme handwriting regressions + independent baseline labeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from domain.entities import BoundingBox
from domain.policy import DecisionEngine, DecisionState
from evaluation.experiment_configs import get_experiment_manifest
from evaluation.independent_baselines import (
    BASELINE_KIND_DERIVED,
    BASELINE_KIND_INDEPENDENT,
    _opencv_line_boxes,
    derived_from_api_result,
)
from infrastructure.models.bbox_validation import validate_bbox


def test_page_sized_detection_rejected():
    bbox = BoundingBox(x=0, y=0, width=580, height=760)
    crop = np.zeros((760, 580), dtype=np.uint8)
    crop[100:700, 50:530] = 40
    vr = validate_bbox(bbox, 600, 800, crop)
    assert vr.reject_for_htr or "excessive_area" in vr.flags or "excessive_height" in vr.flags


def test_bad_crop_low_ink_flagged():
    bbox = BoundingBox(x=10, y=10, width=200, height=28)
    # Blank white crop (no ink) — must not look like a valid line
    crop = np.full((28, 200), 255, dtype=np.uint8)
    vr = validate_bbox(bbox, 600, 800, crop)
    assert "low_ink" in vr.flags or vr.reject_for_htr or not vr.ok


def test_htr_blocked_extreme_abstention():
    result = DecisionEngine().fuse([], htr_blocked=True)
    assert result.state == DecisionState.ILLEGIBLE
    assert result.text == "[ILLEGIBLE]"


def test_accepted_requires_visual_evidence_not_lm_only():
    from domain.entities import Hypothesis
    from domain.policy import normalize_candidate

    # Paddle-only high score on handwriting must not be ACCEPTED
    hyp = Hypothesis(
        recognition_run_id="r",
        text="plausible medical sentence",
        normalized_text=normalize_candidate("plausible medical sentence"),
        rank=1,
        sequence_score=-1,
        visual_score=0.99,
        model_name="paddleocr-rec",
        model_version="1",
        source_crop_id="c",
        image_variant="raw",
        configuration={},
    )
    result = DecisionEngine().fuse([hyp], image_quality_risk=0.25)
    assert result.state != DecisionState.ACCEPTED


def test_opencv_independent_detector_rejects_page_box():
    img = np.full((400, 300, 3), 255, dtype=np.uint8)
    # One giant dark region ≈ page
    img[5:395, 5:295] = 0
    boxes = _opencv_line_boxes(img)
    for x, y, w, h in boxes:
        assert not (w > 0.92 * 300 and h > 0.55 * 400)


def test_derived_baseline_labeled_not_independent():
    fake = {
        "words": [
            {
                "line_id": "L1",
                "reading_order": 0,
                "text": "x",
                "decision_state": "ACCEPTED",
            }
        ],
        "hypotheses_by_line": {
            "L1": [
                {
                    "model_name": "paddleocr-rec",
                    "text": "x",
                    "visual_score": 0.9,
                },
                {
                    "model_name": "microsoft/trocr-large-handwritten",
                    "text": "y",
                    "visual_score": 0.5,
                },
            ]
        },
    }
    _, _, meta = derived_from_api_result(fake)
    assert meta["baseline_kind"] == BASELINE_KIND_DERIVED
    assert meta["baseline_kind"] != BASELINE_KIND_INDEPENDENT
    assert meta.get("shared_detector") is True


def test_config_a_forbids_scribeproof_engine():
    m = get_experiment_manifest("A")
    assert m["baseline_kind"] == "independent"
    assert "DecisionEngine" in m["forbidden_scribeproof_components"]
    assert m["decision_engine"] is False
    assert m["verifier"] is False


def test_example31_image_exists_for_regression():
    root = Path(__file__).resolve().parents[2]
    img = (
        root
        / "evaluation"
        / "dataset"
        / "held_out"
        / "images"
        / "ebe19b27_script_errors_Example31.jpg"
    )
    if not img.is_file():
        pytest.skip("Example31 image not present")
    assert img.stat().st_size > 1000
