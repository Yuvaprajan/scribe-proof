"""Machine-readable ablation configuration manifests (A–F).

Each config differs ONLY by its declared components. Hidden ScribeProof
components must not leak into A/B/C.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


EXPERIMENT_MANIFESTS: dict[str, dict[str, Any]] = {
    "A": {
        "id": "A",
        "name": "independent_paddle_only",
        "baseline_kind": "independent",
        "detector": "paddleocr_det",
        "recognizer": "paddleocr_rec",
        "preprocessing": "none_external",
        "candidate_generation": "paddle_single_hyp",
        "reranking": False,
        "verifier": False,
        "decision_engine": False,
        "thresholds": {"accept_all_non_empty": True},
        "post_processing": "none",
        "correction": False,
        "abstention_behavior": "none",
        "allowed_scribeproof_components": [],
        "uses_scribeproof_components": False,
        "forbidden_scribeproof_components": [
            "LineDetector",
            "DecisionEngine",
            "Verifier",
            "ContextReranker",
            "bbox_sanitize",
            "candidate_pool_fusion",
        ],
    },
    "B": {
        "id": "B",
        "name": "independent_opencv_det_plus_trocr",
        "baseline_kind": "independent",
        "detector": "opencv_morphology_lines",
        "recognizer": "trocr-large-handwritten",
        "preprocessing": "grayscale+otsu_for_detection_only",
        "candidate_generation": "trocr_top1",
        "reranking": False,
        "verifier": False,
        "decision_engine": False,
        "thresholds": {
            "accept": 0.72,
            "review": 0.55,
            "else": "ILLEGIBLE",
        },
        "post_processing": "minimal_threshold_decode",
        "correction": False,
        "abstention_behavior": "score_threshold",
        "allowed_scribeproof_components": ["TrOCRRecognitionProvider"],
        "uses_scribeproof_components": False,
        "forbidden_scribeproof_components": [
            "LineDetector",
            "DecisionEngine",
            "Verifier",
            "ContextReranker",
            "bbox_sanitize",
        ],
        "note": (
            "User-facing ablation label historically mixed sanitized detection; "
            "this config is the independent B (OpenCV lines + TrOCR)."
        ),
    },
    "C": {
        "id": "C",
        "name": "top1_accept_no_decision_engine",
        "baseline_kind": "independent",
        "detector": "opencv_morphology_lines",
        "recognizer": "trocr-large-handwritten",
        "preprocessing": "grayscale+otsu_for_detection_only",
        "candidate_generation": "trocr_top1",
        "reranking": False,
        "verifier": False,
        "decision_engine": False,
        "thresholds": {"force_accept_non_empty": True},
        "post_processing": "force_ACCEPTED_if_non_empty",
        "correction": False,
        "abstention_behavior": "illegible_only_if_empty",
        "inherits_from": "B",
        "diff_from_B": "force ACCEPTED on any non-empty hypothesis",
        "forbidden_scribeproof_components": [
            "DecisionEngine",
            "Verifier",
            "ContextReranker",
        ],
    },
    "D": {
        "id": "D",
        "name": "full_scribeproof_decision_engine",
        "baseline_kind": "system_under_test",
        "detector": "paddle_det_plus_scribeproof_sanitize",
        "recognizer": "paddle_rec_plus_trocr_ensemble",
        "preprocessing": "scribeproof_image_pipeline",
        "candidate_generation": "multi_variant_ensemble",
        "reranking": True,
        "verifier": False,
        "decision_engine": True,
        "thresholds": "DecisionThresholds.default",
        "post_processing": "evidence_fusion",
        "correction": "safe_lexicon_only",
        "abstention_behavior": "decision_engine_states",
        "allowed_scribeproof_components": ["full_pipeline_without_verifier"],
    },
    "E": {
        "id": "E",
        "name": "D_plus_verifier",
        "baseline_kind": "system_under_test",
        "inherits_from": "D",
        "verifier": True,
        "diff_from_D": "verifier selection applied among visual candidates only",
        "decision_engine": True,
        "thresholds": "DecisionThresholds.default",
        "abstention_behavior": "decision_engine_plus_verifier_reject",
    },
    "F": {
        "id": "F",
        "name": "E_plus_conservative_thresholds",
        "baseline_kind": "system_under_test",
        "inherits_from": "E",
        "verifier": True,
        "decision_engine": True,
        "thresholds": "DecisionThresholds.conservative",
        "diff_from_E": "more conservative accept / higher abstention",
        "abstention_behavior": "conservative_thresholds",
    },
}


def get_experiment_manifest(config_id: str) -> dict[str, Any]:
    if config_id not in EXPERIMENT_MANIFESTS:
        raise KeyError(config_id)
    return deepcopy(EXPERIMENT_MANIFESTS[config_id])


def all_experiment_manifests() -> dict[str, dict[str, Any]]:
    return deepcopy(EXPERIMENT_MANIFESTS)
