"""Layout intelligence: PaddleOCR text-line detection + crossed-out candidates."""

from __future__ import annotations

# Ensure torch is loaded before paddle on Windows when both are used in-process.
try:
    import torch  # noqa: F401
except Exception:
    pass

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from domain.entities import BoundingBox, RegionType

logger = logging.getLogger(__name__)


@dataclass
class DetectedLine:
    bbox: BoundingBox
    polygon: list[list[float]]
    confidence: float
    region_type: RegionType = RegionType.MAIN_HANDWRITING
    is_crossed_out_candidate: bool = False
    crossed_out_score: float = 0.0
    paddle_text: Optional[str] = None
    paddle_score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    lines: list[DetectedLine]
    reading_order_edges: list[tuple[int, int]]
    detector_name: str
    settings: dict[str, Any] = field(default_factory=dict)


class LineDetector:
    """PaddleOCR PP-OCR detection (preferred) with OpenCV morphology fallback."""

    def __init__(self, use_paddle: bool = True):
        self.use_paddle = use_paddle
        self._ocr = None
        self._init_failed = False
        self._init_error: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self._ocr is not None

    @property
    def status(self) -> dict[str, Any]:
        return {
            "provider": "paddleocr",
            "enabled": self.use_paddle,
            "ready": self.ready,
            "init_failed": self._init_failed,
            "error": self._init_error,
        }

    def warmup(self) -> bool:
        return self._get_paddle() is not None

    def _get_paddle(self):
        if self._ocr is not None:
            return self._ocr
        if self._init_failed or not self.use_paddle:
            return None
        try:
            from paddleocr import PaddleOCR

            # Prefer server detection models when available (PP-OCRv4/v5).
            # ocr_version selects the detection/recognition stack.
            ocr_version = os.getenv("SCRIBEPROOF_PADDLE_OCR_VERSION", "PP-OCRv4")
            kwargs: dict[str, Any] = {
                "use_angle_cls": True,
                "lang": os.getenv("SCRIBEPROOF_PADDLE_LANG", "en"),
                "show_log": False,
                "use_gpu": os.getenv("SCRIBEPROOF_PADDLE_GPU", "false").lower()
                in {"1", "true", "yes"},
            }
            # paddleocr 2.7+ supports ocr_version
            try:
                self._ocr = PaddleOCR(ocr_version=ocr_version, **kwargs)
            except TypeError:
                self._ocr = PaddleOCR(**kwargs)

            logger.info(
                "PaddleOCR detector initialized (ocr_version=%s)", ocr_version
            )
            return self._ocr
        except Exception as e:
            self._init_failed = True
            self._init_error = str(e)
            logger.warning("PaddleOCR unavailable (%s); using OpenCV fallback", e)
            return None

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        paddle = self._get_paddle()
        if paddle is not None:
            try:
                return self._detect_paddle(image_bgr, paddle)
            except Exception as e:
                logger.warning("Paddle detection failed (%s); OpenCV fallback", e)
        return self._detect_opencv(image_bgr)

    def _poly_to_bbox(self, poly: list) -> BoundingBox:
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return BoundingBox(x=x0, y=y0, width=max(1.0, x1 - x0), height=max(1.0, y1 - y0))

    def _normalize_paddle_result(self, result: Any) -> list[Any]:
        """Normalize PaddleOCR output across 2.x result shapes."""
        if result is None:
            return []
        # Common: [ [ [box, (text, score)], ... ] ]
        if isinstance(result, list) and result and result[0] is None:
            return []
        if isinstance(result, list) and result and isinstance(result[0], list):
            # page-level list
            page = result[0]
            if page and isinstance(page[0], dict):
                # newer dict format
                lines = []
                for item in page:
                    box = item.get("dt_polys") or item.get("box") or item.get("poly")
                    txt = item.get("rec_txt") or item.get("text") or ""
                    score = float(item.get("rec_score") or item.get("score") or 0.5)
                    if box is not None:
                        lines.append([box, (txt, score)])
                return lines
            return page
        return []

    def _detect_paddle(self, image_bgr: np.ndarray, ocr) -> DetectionResult:
        # Prefer detection boxes; recognition text is only a weak secondary hint.
        # Handwriting transcription is TrOCR's job — Paddle print OCR is unreliable.
        result = None
        try:
            result = ocr.ocr(image_bgr, det=True, rec=True, cls=True)
        except TypeError:
            try:
                result = ocr.ocr(image_bgr, cls=True)
            except TypeError:
                result = ocr.ocr(image_bgr)

        items = self._normalize_paddle_result(result)
        lines: list[DetectedLine] = []
        if not items:
            logger.info("PaddleOCR returned no lines; falling back to OpenCV")
            return self._detect_opencv(image_bgr)

        h, w = image_bgr.shape[:2]
        for item in items:
            try:
                poly = item[0]
                text_info = item[1] if len(item) > 1 else ("", 0.5)
                if isinstance(text_info, (list, tuple)):
                    paddle_text = str(text_info[0]) if text_info else ""
                    conf = float(text_info[1]) if len(text_info) > 1 else 0.5
                else:
                    paddle_text = str(text_info)
                    conf = 0.5
            except Exception:
                continue

            # poly may be ndarray
            poly_list = [[float(p[0]), float(p[1])] for p in poly]
            bbox = self._poly_to_bbox(poly_list)
            if bbox.width < 8 or bbox.height < 6:
                continue

            crop = image_bgr[
                max(0, int(bbox.y)) : min(h, int(bbox.y + bbox.height)),
                max(0, int(bbox.x)) : min(w, int(bbox.x + bbox.width)),
            ]
            crossed, score = self._crossed_out_score(crop)
            script_mode, stroke_irreg = self._script_mode(crop, conf)
            region_type = self._classify_region(
                bbox, w, h, crossed, script_mode=script_mode
            )
            # Margin / signature regions: require stronger cancellation evidence
            if crossed and region_type in {
                RegionType.MARGIN_NOTE,
                RegionType.SIGNATURE,
            }:
                if score < 0.8:
                    crossed = False
                    region_type = self._classify_region(
                        bbox, w, h, False, script_mode=script_mode
                    )
            lines.append(
                DetectedLine(
                    bbox=bbox,
                    polygon=poly_list,
                    confidence=conf,
                    region_type=region_type,
                    is_crossed_out_candidate=crossed,
                    crossed_out_score=score,
                    paddle_text=paddle_text.strip() or None,
                    paddle_score=conf if paddle_text.strip() else None,
                    metadata={
                        "paddle_text": paddle_text,
                        "paddle_score": conf,
                        "script_mode": script_mode,
                        "stroke_irregularity": stroke_irreg,
                    },
                )
            )

        if not lines:
            return self._detect_opencv(image_bgr)

        lines = self._assign_reading_order(lines)
        edges = [(i, i + 1) for i in range(len(lines) - 1)]
        return DetectionResult(
            lines=lines,
            reading_order_edges=edges,
            detector_name="paddleocr",
            settings={
                "engine": "PaddleOCR PP-OCR detection",
                "ocr_version": os.getenv("SCRIBEPROOF_PADDLE_OCR_VERSION", "PP-OCRv4"),
            },
        )

    def _detect_opencv(self, image_bgr: np.ndarray) -> DetectionResult:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = image_bgr.shape[:2]
        lines: list[DetectedLine] = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 20 or bh < 8 or bw * bh < 200:
                continue
            aspect = bw / max(bh, 1)
            if aspect < 1.2 and bh < 40:
                continue
            bbox = BoundingBox(x=float(x), y=float(y), width=float(bw), height=float(bh))
            crop = image_bgr[y : y + bh, x : x + bw]
            crossed, score = self._crossed_out_score(crop)
            script_mode, stroke_irreg = self._script_mode(crop, 0.55)
            region_type = self._classify_region(
                bbox, w, h, crossed, script_mode=script_mode
            )
            poly = [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]
            lines.append(
                DetectedLine(
                    bbox=bbox,
                    polygon=[[float(a), float(b)] for a, b in poly],
                    confidence=0.55,
                    region_type=region_type,
                    is_crossed_out_candidate=crossed,
                    crossed_out_score=score,
                    metadata={
                        "script_mode": script_mode,
                        "stroke_irregularity": stroke_irreg,
                    },
                )
            )
        lines = self._assign_reading_order(lines)
        edges = [(i, i + 1) for i in range(len(lines) - 1)]
        return DetectionResult(
            lines=lines,
            reading_order_edges=edges,
            detector_name="opencv_fallback",
            settings={"method": "adaptive_threshold_morphology"},
        )

    def _script_mode(self, crop: np.ndarray, paddle_conf: float) -> tuple[str, float]:
        """Estimate print vs handwriting from stroke irregularity + Paddle confidence."""
        if crop is None or crop.size == 0:
            return "handwriting", 0.5
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        if gray.shape[0] < 6 or gray.shape[1] < 12:
            return "handwriting", 0.5
        # Vertical projection variance of ink thickness — print is more uniform
        ink = (gray < 170).astype(np.float32)
        col_sums = ink.sum(axis=0)
        active = col_sums[col_sums > 0]
        if active.size < 5:
            irreg = 0.5
        else:
            irreg = float(np.std(active) / (np.mean(active) + 1e-3))
            irreg = min(1.0, irreg / 8.0)
        print_primary = os.getenv("SCRIBEPROOF_PRINT_OCR_PRIMARY", "true").lower() in {
            "1",
            "true",
            "yes",
        }
        # Strict print gate: high Paddle conf AND uniform strokes.
        # Ambiguous / messy ink defaults to handwriting (TrOCR-primary).
        if print_primary and paddle_conf >= 0.88 and irreg < 0.22:
            return "print", irreg
        if print_primary and paddle_conf >= 0.95 and irreg < 0.30:
            return "print", irreg
        if irreg > 0.35 or paddle_conf < 0.60:
            return "handwriting", irreg
        return "handwriting", irreg

    def _classify_region(
        self,
        bbox: BoundingBox,
        page_w: int,
        page_h: int,
        crossed: bool,
        *,
        script_mode: str = "handwriting",
    ) -> RegionType:
        if crossed:
            return RegionType.CROSSED_OUT_CANDIDATE
        if bbox.x < 0.12 * page_w or (bbox.x + bbox.width) > 0.88 * page_w:
            if bbox.width < 0.35 * page_w:
                return RegionType.MARGIN_NOTE
        if bbox.y > 0.82 * page_h and bbox.height < 0.08 * page_h:
            return RegionType.SIGNATURE
        if script_mode == "print":
            return RegionType.PRINTED_TEXT
        return RegionType.MAIN_HANDWRITING

    def _crossed_out_score(self, crop: np.ndarray) -> tuple[bool, float]:
        if crop is None or crop.size == 0:
            return False, 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        if gray.shape[0] < 8 or gray.shape[1] < 20:
            return False, 0.0
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=int(0.45 * gray.shape[1]),
            maxLineGap=8,
        )
        if lines is None:
            return False, 0.0
        ink = (gray < 180).astype(np.uint8)
        ink_pixels = max(int(ink.sum()), 1)
        stroke_mask = np.zeros_like(gray)
        long_strokes = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if length >= 0.4 * gray.shape[1] and (
                angle < 25 or 35 < angle < 55 or angle > 155
            ):
                long_strokes += 1
                cv2.line(stroke_mask, (x1, y1), (x2, y2), 255, 2)
        if long_strokes == 0:
            return False, 0.0
        intersection = np.logical_and(stroke_mask > 0, ink > 0).sum()
        density = float(intersection) / float(ink_pixels)
        score = min(1.0, 0.35 * long_strokes + 4.0 * density)
        return score >= 0.55, score

    def _assign_reading_order(self, lines: list[DetectedLine]) -> list[DetectedLine]:
        sorted_lines = sorted(
            lines, key=lambda L: (L.bbox.y + L.bbox.height / 2, L.bbox.x)
        )
        body = [
            L
            for L in sorted_lines
            if L.region_type
            in {RegionType.MAIN_HANDWRITING, RegionType.PRINTED_TEXT, RegionType.TABLE}
        ]
        for i, L in enumerate(sorted_lines):
            L.metadata["reading_order"] = i
            if L.region_type == RegionType.MARGIN_NOTE and body:
                cy = L.bbox.y + L.bbox.height / 2
                nearest = min(
                    body, key=lambda b: abs((b.bbox.y + b.bbox.height / 2) - cy)
                )
                L.metadata["linked_body_y"] = nearest.bbox.y
        return sorted_lines
