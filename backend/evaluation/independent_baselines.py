"""Independent external baselines (NOT derived from ScribeProof).

Independent Paddle-only
-----------------------
  raw image → PaddleOCR detection + recognition → ACCEPTED all lines
  Shared preprocessing: none beyond what PaddleOCR applies internally.
  Does NOT use: ScribeProof LineDetector sanitize, decision engine, verifier,
  candidate pool, routing, correction, or confidence fusion.

Independent TrOCR-only
----------------------
  raw image → independent OpenCV morphology line crops → TrOCR →
  minimal threshold decode (accept / review / illegible).
  Shared preprocessing (documented): grayscale + Otsu binarize for detection
  only; recognition uses the matching raw RGB crop (no CLAHE/deskew from
  ScribeProof ImagePipeline).
  Does NOT use: ScribeProof LineDetector, bbox sanitize/split, decision engine,
  verifier, Paddle detections, or lexicon correction.

Derived baselines (ablation-only)
---------------------------------
  Same detections/crops as a ScribeProof API run. Must be labeled
  ``baseline_kind: derived`` — never ``independent``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger("independent_baselines")

BASELINE_KIND_INDEPENDENT = "independent"
BASELINE_KIND_DERIVED = "derived"
BASELINE_KIND_SHARED_DETECTOR = "shared_detector"


def independent_paddle_only(image_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Raw image → Paddle det+rec. All non-empty lines ACCEPTED."""
    meta = {
        "baseline_kind": BASELINE_KIND_INDEPENDENT,
        "label": "independent_paddle_only",
        "detector": "paddleocr_det",
        "recognizer": "paddleocr_rec",
        "preprocessing": "none_external",
        "shared_preprocessing": [],
        "uses_scribeproof_components": False,
        "decision_engine": False,
        "verifier": False,
        "correction": False,
        "abstention": "none_all_accepted",
    }
    try:
        from paddleocr import PaddleOCR
    except Exception as e:
        logger.warning("Paddle unavailable: %s", e)
        return [], {**meta, "error": str(e)}

    img = cv2.imread(str(image_path))
    if img is None:
        return [], {**meta, "error": "imread_failed"}

    t0 = time.time()
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    try:
        result = ocr.ocr(img, cls=True)
    except TypeError:
        result = ocr.ocr(img)
    page = result[0] if result else []
    words: list[dict[str, Any]] = []
    for i, item in enumerate(page or []):
        try:
            text = str(item[1][0])
            score = float(item[1][1])
        except Exception:
            continue
        if not text.strip():
            continue
        words.append(
            {
                "reference": "",
                "hypothesis": text,
                "decision_state": "ACCEPTED",
                "confidence": score,
                "reading_order": i,
                "model_score": score,
                "uncertainty_score": max(0.0, 1.0 - score),
                "calibrated_confidence": None,
                "confidence_is_calibrated": False,
            }
        )
    meta["elapsed_seconds"] = time.time() - t0
    meta["n_lines"] = len(words)
    return words, meta


def _opencv_line_boxes(img_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Minimal independent line detector (morphology). Not ScribeProof."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Horizontal dilation to connect characters into lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 40), 3))
    merged = cv2.dilate(bw, kernel, iterations=1)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, bw_, bh_ = cv2.boundingRect(c)
        area = bw_ * bh_
        if area < 80 or bh_ < 8 or bw_ < 20:
            continue
        # Reject near page-sized boxes (independent of ScribeProof sanitize)
        if bw_ > 0.92 * w and bh_ > 0.55 * h:
            continue
        if bh_ > 0.35 * h and bw_ > 0.85 * w:
            continue
        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + bw_ + pad)
        y1 = min(h, y + bh_ + pad)
        boxes.append((x0, y0, x1 - x0, y1 - y0))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def independent_trocr_only(image_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Raw image → OpenCV lines → TrOCR. No ScribeProof detector/engine."""
    import sys

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    meta = {
        "baseline_kind": BASELINE_KIND_INDEPENDENT,
        "label": "independent_trocr_only",
        "detector": "opencv_morphology_lines",
        "recognizer": "microsoft/trocr-large-handwritten",
        "preprocessing": "grayscale+otsu_for_detection_only",
        "shared_preprocessing": [
            "grayscale",
            "gaussian_blur_3x3",
            "otsu_binarize_for_line_find",
        ],
        "uses_scribeproof_components": False,
        "decision_engine": False,
        "verifier": False,
        "correction": False,
        "abstention": "score_threshold_0.55_illegible_0.72_accept",
        "note": (
            "Recognition crops are raw BGR sub-images. Detection uses Otsu morphology "
            "only — not ScribeProof LineDetector, CLAHE, deskew, or bbox sanitize."
        ),
    }
    img = cv2.imread(str(image_path))
    if img is None:
        return [], {**meta, "error": "imread_failed"}

    try:
        from infrastructure.models.recognition import TrOCRRecognitionProvider
    except Exception as e:
        logger.warning("TrOCR unavailable: %s", e)
        return [], {**meta, "error": str(e), "baseline_kind": BASELINE_KIND_INDEPENDENT}

    t0 = time.time()
    boxes = _opencv_line_boxes(img)
    try:
        rec = TrOCRRecognitionProvider()
    except Exception as e:
        logger.warning("TrOCR provider init failed: %s", e)
        return [], {
            **meta,
            "error": str(e),
            "note": (
                "Independent TrOCR could not load (often WinError 127 / torch DLL "
                "conflict when the API already holds GPU). Re-run with API stopped, "
                "or use --derived-only and label results as derived — never independent."
            ),
        }

    words: list[dict[str, Any]] = []
    load_failed = False
    for i, (x, y, bw, bh) in enumerate(boxes):
        crop = img[y : y + bh, x : x + bw]
        if crop.size == 0:
            continue
        try:
            hyps = rec.recognize(crop, {"image_variant": "raw", "fast": True})
        except Exception as e:
            logger.warning("TrOCR recognize failed: %s", e)
            load_failed = True
            break
        text = hyps[0].text if hyps else ""
        score = float(hyps[0].visual_score) if hyps else 0.0
        if not text.strip():
            state = "ILLEGIBLE"
            hyp = "[ILLEGIBLE]"
        elif score >= 0.72:
            state = "ACCEPTED"
            hyp = text
        elif score >= 0.55:
            state = "REVIEW_REQUIRED"
            hyp = text
        else:
            state = "ILLEGIBLE"
            hyp = "[ILLEGIBLE]"
        words.append(
            {
                "reference": "",
                "hypothesis": hyp,
                "decision_state": state,
                "confidence": score,
                "reading_order": i,
                "model_score": score,
                "uncertainty_score": max(0.0, 1.0 - score),
                "calibrated_confidence": None,
                "confidence_is_calibrated": False,
            }
        )
    meta["elapsed_seconds"] = time.time() - t0
    meta["n_lines"] = len(words)
    produced_text = any(
        (w.get("hypothesis") or "") not in {"", "[ILLEGIBLE]"} for w in words
    )
    if load_failed or (boxes and words and not produced_text):
        meta["error"] = meta.get("error") or "trocr_runtime_unavailable"
        meta["note"] = (
            "Independent TrOCR produced no readable text (model load/runtime "
            "failure is common when another process holds torch/CUDA). "
            "Do not treat empty-ILLEGIBLE output as a valid TrOCR baseline; "
            "re-run without the API process, or use a derived baseline "
            "explicitly labeled baseline_kind=derived."
        )
        # Return empty so callers can fall back without publishing fake metrics
        return [], meta
    return words, meta


def derived_from_api_result(
    result: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Ablation-only derived baselines from one ScribeProof API result.

    Labels: baseline_kind=derived, shared_detector=true.
    """
    meta = {
        "baseline_kind": BASELINE_KIND_DERIVED,
        "shared_detector": True,
        "label": "derived_from_scribeproof_api_run",
        "warning": (
            "NOT an independent external baseline. Same detections/crops as "
            "ScribeProof. Use only for internal ablation / same-input comparison."
        ),
        "uses_scribeproof_components": True,
    }

    def _best(hyps: list[dict], family: str) -> Optional[dict]:
        scored = []
        for h in hyps or []:
            name = (h.get("model_name") or "").lower()
            if family == "paddle" and "paddle" in name:
                scored.append(h)
            elif family == "trocr" and ("trocr" in name or "microsoft" in name):
                scored.append(h)
        if not scored:
            return None
        return max(scored, key=lambda h: float(h.get("visual_score") or 0.0))

    hyps_by_line = result.get("hypotheses_by_line") or {}
    words = sorted(
        result.get("words") or [], key=lambda w: int(w.get("reading_order") or 0)
    )
    paddle_words: list[dict[str, Any]] = []
    trocr_words: list[dict[str, Any]] = []
    for w in words:
        line_id = w.get("line_id") or ""
        hyps = hyps_by_line.get(line_id) or []
        order = int(w.get("reading_order") or 0)
        p = _best(hyps, "paddle")
        if p and (p.get("text") or "").strip():
            paddle_words.append(
                {
                    "reference": "",
                    "hypothesis": p["text"],
                    "decision_state": "ACCEPTED",
                    "confidence": float(p.get("visual_score") or 0.0),
                    "reading_order": order,
                }
            )
        else:
            paddle_words.append(
                {
                    "reference": "",
                    "hypothesis": "[ILLEGIBLE]",
                    "decision_state": "ILLEGIBLE",
                    "confidence": 0.0,
                    "reading_order": order,
                }
            )
        t = _best(hyps, "trocr")
        if t and (t.get("text") or "").strip():
            score = float(t.get("visual_score") or 0.0)
            state = "ACCEPTED" if score >= 0.72 else "REVIEW_REQUIRED"
            trocr_words.append(
                {
                    "reference": "",
                    "hypothesis": t["text"],
                    "decision_state": state,
                    "confidence": score,
                    "reading_order": order,
                }
            )
        else:
            trocr_words.append(
                {
                    "reference": "",
                    "hypothesis": "[ILLEGIBLE]",
                    "decision_state": "ILLEGIBLE",
                    "confidence": 0.0,
                    "reading_order": order,
                }
            )
    return paddle_words, trocr_words, meta
