"""Detection bbox validation + secondary line splitting (P0 safety).

A page-sized or multi-line region must NEVER be sent to line-level TrOCR.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from domain.entities import BoundingBox, RegionType

logger = logging.getLogger(__name__)


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass
class BBoxThresholds:
    max_area_ratio: float = field(default_factory=lambda: _f("SCRIBEPROOF_BBOX_MAX_AREA_RATIO", 0.22))
    max_height_ratio: float = field(default_factory=lambda: _f("SCRIBEPROOF_BBOX_MAX_HEIGHT_RATIO", 0.18))
    max_aspect_tall: float = field(default_factory=lambda: _f("SCRIBEPROOF_BBOX_MAX_ASPECT_TALL", 0.55))
    min_line_height_px: float = field(default_factory=lambda: _f("SCRIBEPROOF_BBOX_MIN_HEIGHT_PX", 8.0))
    max_line_height_px: float = field(default_factory=lambda: _f("SCRIBEPROOF_BBOX_MAX_HEIGHT_PX", 140.0))
    min_width_px: float = field(default_factory=lambda: _f("SCRIBEPROOF_BBOX_MIN_WIDTH_PX", 12.0))
    min_ink_ratio: float = field(default_factory=lambda: _f("SCRIBEPROOF_BBOX_MIN_INK_RATIO", 0.004))
    max_projected_lines: int = field(
        default_factory=lambda: int(os.getenv("SCRIBEPROOF_BBOX_MAX_PROJECTED_LINES", "2"))
    )
    tiny_upscale_limit: float = field(default_factory=lambda: _f("SCRIBEPROOF_TROCR_MAX_UPSCALE", 3.5))
    trocr_target_h: int = field(
        default_factory=lambda: int(os.getenv("SCRIBEPROOF_TROCR_TARGET_H", "80"))
    )


@dataclass
class ValidationResult:
    ok: bool
    reason: str
    flags: list[str] = field(default_factory=list)
    estimated_lines: int = 1
    area_ratio: float = 0.0
    height_ratio: float = 0.0
    ink_ratio: float = 0.0
    needs_split: bool = False
    reject_for_htr: bool = False


def ink_ratio(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    ink = (gray < 170).astype(np.float32)
    return float(ink.mean())


def estimate_line_count(crop: np.ndarray) -> int:
    """Horizontal projection valleys → approximate baseline count."""
    if crop is None or crop.size == 0:
        return 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    h, w = gray.shape[:2]
    if h < 12 or w < 12:
        return 1
    # Adaptive: ink rows
    thr = np.percentile(gray, 55)
    ink = (gray < thr).astype(np.uint8)
    proj = ink.sum(axis=1).astype(np.float32)
    if proj.max() < 1:
        return 0
    # Smooth
    k = max(3, h // 40)
    if k % 2 == 0:
        k += 1
    proj_s = cv2.GaussianBlur(proj.reshape(-1, 1), (1, k), 0).ravel()
    thresh = 0.28 * float(proj_s.max())
    active = proj_s > thresh
    # Count contiguous active bands
    bands = 0
    in_band = False
    for v in active:
        if v and not in_band:
            bands += 1
            in_band = True
        elif not v:
            in_band = False
    return max(1, bands) if bands else 0


def validate_bbox(
    bbox: BoundingBox,
    page_w: int,
    page_h: int,
    crop: Optional[np.ndarray] = None,
    thresholds: Optional[BBoxThresholds] = None,
) -> ValidationResult:
    th = thresholds or BBoxThresholds()
    page_area = max(page_w * page_h, 1)
    area = max(bbox.width * bbox.height, 1.0)
    area_ratio = area / page_area
    height_ratio = bbox.height / max(page_h, 1)
    aspect = bbox.width / max(bbox.height, 1.0)  # width/height; tall ⇒ small
    flags: list[str] = []

    ink = ink_ratio(crop) if crop is not None else 0.0
    n_lines = estimate_line_count(crop) if crop is not None else 1

    if area_ratio > th.max_area_ratio:
        flags.append("excessive_area")
    if height_ratio > th.max_height_ratio:
        flags.append("excessive_height")
    if bbox.height > th.max_line_height_px and aspect < 2.0:
        flags.append("tall_box")
    if aspect < th.max_aspect_tall and bbox.height > 40:
        flags.append("implausible_aspect")
    if bbox.height < th.min_line_height_px or bbox.width < th.min_width_px:
        flags.append("too_tiny")
    if crop is not None and ink < th.min_ink_ratio:
        flags.append("low_ink")
    if n_lines > th.max_projected_lines:
        flags.append("multi_line_projection")

    needs_split = bool(
        {"excessive_area", "excessive_height", "tall_box", "multi_line_projection", "implausible_aspect"}
        & set(flags)
    )
    # Page-like: hard reject for direct HTR
    reject_for_htr = (
        area_ratio > th.max_area_ratio
        or height_ratio > th.max_height_ratio
        or "multi_line_projection" in flags
        or "excessive_area" in flags
    )
    # Tiny with destructive upscale
    if crop is not None and crop.shape[0] > 0:
        upscale = th.trocr_target_h / float(crop.shape[0])
        if upscale > th.tiny_upscale_limit:
            flags.append("destructive_upscale")
            reject_for_htr = True

    ok = not flags or (
        not needs_split
        and "too_tiny" not in flags
        and "low_ink" not in flags
        and "destructive_upscale" not in flags
    )
    if needs_split:
        ok = False
    reason = "valid_line" if ok else ",".join(flags) if flags else "invalid"
    return ValidationResult(
        ok=ok,
        reason=reason,
        flags=flags,
        estimated_lines=n_lines,
        area_ratio=area_ratio,
        height_ratio=height_ratio,
        ink_ratio=ink,
        needs_split=needs_split,
        reject_for_htr=reject_for_htr,
    )


def validate_trocr_crop(
    crop: np.ndarray,
    *,
    page_w: Optional[int] = None,
    page_h: Optional[int] = None,
    thresholds: Optional[BBoxThresholds] = None,
) -> ValidationResult:
    """Contract check before prepare_trocr_image / TrOCR inference."""
    th = thresholds or BBoxThresholds()
    if crop is None or crop.size == 0:
        return ValidationResult(
            ok=False, reason="empty_crop", flags=["empty"], reject_for_htr=True
        )
    h, w = crop.shape[:2]
    if page_w is None:
        page_w = max(w * 5, w)
    if page_h is None:
        page_h = max(h * 5, h)
    bbox = BoundingBox(x=0, y=0, width=float(w), height=float(h))
    # Relative to crop-as-page is weak; use absolute pixel rules for crops
    flags: list[str] = []
    n_lines = estimate_line_count(crop)
    ink = ink_ratio(crop)
    if h > th.max_line_height_px:
        flags.append("crop_too_tall")
    if n_lines > th.max_projected_lines:
        flags.append("multi_line_projection")
    if ink < th.min_ink_ratio:
        flags.append("low_ink")
    if h < th.min_line_height_px or w < th.min_width_px:
        flags.append("too_tiny")
    upscale = th.trocr_target_h / float(max(h, 1))
    if upscale > th.tiny_upscale_limit:
        flags.append("destructive_upscale")
    # Page-fraction if parent page known and crop almost page-sized
    area_ratio = (w * h) / max(page_w * page_h, 1)
    if area_ratio > th.max_area_ratio:
        flags.append("excessive_area")
    reject = bool(
        flags
        and any(
            f in flags
            for f in (
                "crop_too_tall",
                "multi_line_projection",
                "excessive_area",
                "destructive_upscale",
                "low_ink",
                "too_tiny",
            )
        )
    )
    # Allow mild low-ink / slightly short crops with review path upstream
    ok = not reject or (
        flags == ["too_tiny"] and h >= 6 and ink >= th.min_ink_ratio * 0.5
    )
    if "excessive_area" in flags or "multi_line_projection" in flags or "crop_too_tall" in flags:
        ok = False
        reject = True
    return ValidationResult(
        ok=ok and not reject,
        reason="valid_line" if (ok and not reject) else ",".join(flags),
        flags=flags,
        estimated_lines=n_lines,
        area_ratio=area_ratio,
        height_ratio=h / max(page_h, 1),
        ink_ratio=ink,
        needs_split="multi_line_projection" in flags or "crop_too_tall" in flags,
        reject_for_htr=reject,
    )


def split_region_horizontal(
    image_bgr: np.ndarray,
    bbox: BoundingBox,
    *,
    parent_meta: Optional[dict[str, Any]] = None,
) -> list[tuple[BoundingBox, dict[str, Any]]]:
    """Projection-profile split of a tall/merged region into child line boxes."""
    h_img, w_img = image_bgr.shape[:2]
    x0 = max(0, int(bbox.x))
    y0 = max(0, int(bbox.y))
    x1 = min(w_img, int(bbox.x + bbox.width))
    y1 = min(h_img, int(bbox.y + bbox.height))
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return []

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ch, cw = gray.shape[:2]
    thr = np.percentile(gray, 55)
    ink = (gray < thr).astype(np.uint8)
    proj = ink.sum(axis=1).astype(np.float32)
    if proj.max() < 3:
        return []

    k = max(3, ch // 50)
    if k % 2 == 0:
        k += 1
    proj_s = cv2.GaussianBlur(proj.reshape(-1, 1), (1, k), 0).ravel()
    thresh = 0.22 * float(proj_s.max())
    active = proj_s > thresh

    bands: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(active):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= 6:
                bands.append((start, i))
            start = None
    if start is not None and ch - start >= 6:
        bands.append((start, ch))

    # Merge bands that are too close (same line)
    merged: list[tuple[int, int]] = []
    for b in bands:
        if not merged:
            merged.append(b)
            continue
        ps, pe = merged[-1]
        if b[0] - pe < max(4, ch // 80):
            merged[-1] = (ps, b[1])
        else:
            merged.append(b)

    if len(merged) < 2:
        # Morphology fallback: horizontal close then find contours
        return _split_morphology(image_bgr, bbox, parent_meta)

    children: list[tuple[BoundingBox, dict[str, Any]]] = []
    for i, (ys, ye) in enumerate(merged):
        pad = 2
        cy0 = max(0, ys - pad)
        cy1 = min(ch, ye + pad)
        # Trim horizontally to ink
        band = ink[cy0:cy1, :]
        cols = band.sum(axis=0)
        nz = np.where(cols > 0)[0]
        if nz.size < 4:
            continue
        cx0 = max(0, int(nz[0]) - 2)
        cx1 = min(cw, int(nz[-1]) + 2)
        child = BoundingBox(
            x=float(x0 + cx0),
            y=float(y0 + cy0),
            width=float(max(1, cx1 - cx0)),
            height=float(max(1, cy1 - cy0)),
        )
        meta = {
            "split_from": "projection_profile",
            "parent_bbox": bbox.as_dict(),
            "child_index": i,
            **(parent_meta or {}),
        }
        children.append((child, meta))
    return children if len(children) >= 2 else _split_morphology(image_bgr, bbox, parent_meta)


def _split_morphology(
    image_bgr: np.ndarray,
    bbox: BoundingBox,
    parent_meta: Optional[dict[str, Any]],
) -> list[tuple[BoundingBox, dict[str, Any]]]:
    h_img, w_img = image_bgr.shape[:2]
    x0 = max(0, int(bbox.x))
    y0 = max(0, int(bbox.y))
    x1 = min(w_img, int(bbox.x + bbox.width))
    y1 = min(h_img, int(bbox.y + bbox.height))
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, crop.shape[1] // 20), 2))
    merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    children: list[tuple[BoundingBox, dict[str, Any]]] = []
    for i, cnt in enumerate(contours):
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 16 or bh < 8:
            continue
        child = BoundingBox(
            x=float(x0 + x),
            y=float(y0 + y),
            width=float(bw),
            height=float(bh),
        )
        children.append(
            (
                child,
                {
                    "split_from": "morphology",
                    "parent_bbox": bbox.as_dict(),
                    "child_index": i,
                    **(parent_meta or {}),
                },
            )
        )
    children.sort(key=lambda t: (t[0].y, t[0].x))
    return children


def sanitize_detection_lines(
    lines: list[Any],
    image_bgr: np.ndarray,
    *,
    thresholds: Optional[BBoxThresholds] = None,
) -> list[Any]:
    """Validate + split suspicious DetectedLine objects in-place list rebuild.

    Expects objects with .bbox, .polygon, .metadata, .region_type, etc.
    Imported lazily-compatible with detection.DetectedLine.
    """
    from infrastructure.models.detection import DetectedLine

    th = thresholds or BBoxThresholds()
    page_h, page_w = image_bgr.shape[:2]
    out: list[DetectedLine] = []

    for line in lines:
        x0 = max(0, int(line.bbox.x))
        y0 = max(0, int(line.bbox.y))
        x1 = min(page_w, int(line.bbox.x + line.bbox.width))
        y1 = min(page_h, int(line.bbox.y + line.bbox.height))
        crop = image_bgr[y0:y1, x0:x1]
        vr = validate_bbox(line.bbox, page_w, page_h, crop, th)
        meta = dict(line.metadata or {})
        meta["bbox_validation"] = {
            "ok": vr.ok,
            "reason": vr.reason,
            "flags": vr.flags,
            "estimated_lines": vr.estimated_lines,
            "area_ratio": vr.area_ratio,
            "height_ratio": vr.height_ratio,
            "ink_ratio": vr.ink_ratio,
            "needs_split": vr.needs_split,
            "reject_for_htr": vr.reject_for_htr,
        }

        if vr.ok and not vr.needs_split:
            line.metadata = meta
            out.append(line)
            continue

        if vr.needs_split or vr.reject_for_htr:
            children = split_region_horizontal(
                image_bgr, line.bbox, parent_meta={"parent_flags": vr.flags}
            )
            if children:
                for child_bbox, child_meta in children:
                    cx0 = max(0, int(child_bbox.x))
                    cy0 = max(0, int(child_bbox.y))
                    cx1 = min(page_w, int(child_bbox.x + child_bbox.width))
                    cy1 = min(page_h, int(child_bbox.y + child_bbox.height))
                    ccrop = image_bgr[cy0:cy1, cx0:cx1]
                    cvr = validate_bbox(child_bbox, page_w, page_h, ccrop, th)
                    # Skip children that are still page-like
                    if cvr.reject_for_htr and cvr.area_ratio > th.max_area_ratio:
                        continue
                    poly = [
                        [child_bbox.x, child_bbox.y],
                        [child_bbox.x + child_bbox.width, child_bbox.y],
                        [
                            child_bbox.x + child_bbox.width,
                            child_bbox.y + child_bbox.height,
                        ],
                        [child_bbox.x, child_bbox.y + child_bbox.height],
                    ]
                    child_line = DetectedLine(
                        bbox=child_bbox,
                        polygon=poly,
                        confidence=line.confidence * 0.9,
                        region_type=line.region_type
                        if line.region_type != RegionType.CROSSED_OUT_CANDIDATE
                        else RegionType.MAIN_HANDWRITING,
                        is_crossed_out_candidate=False,
                        crossed_out_score=0.0,
                        paddle_text=None,
                        paddle_score=None,
                        metadata={
                            **{k: v for k, v in meta.items() if k != "bbox_validation"},
                            "bbox_validation": {
                                "ok": cvr.ok,
                                "reason": cvr.reason,
                                "flags": cvr.flags,
                                "estimated_lines": cvr.estimated_lines,
                                "area_ratio": cvr.area_ratio,
                                "parent_split": True,
                            },
                            "split_meta": child_meta,
                            "script_mode": meta.get("script_mode", "handwriting"),
                        },
                    )
                    out.append(child_line)
                continue

            # Could not split: keep with reject flag for pipeline to abstain
            meta["htr_blocked"] = True
            meta["htr_block_reason"] = vr.reason
            line.metadata = meta
            # Do not forward page-sized boxes to recognition
            if vr.area_ratio > th.max_area_ratio or vr.height_ratio > th.max_height_ratio:
                logger.warning(
                    "Dropping page-like detection box area=%.2f height=%.2f flags=%s",
                    vr.area_ratio,
                    vr.height_ratio,
                    vr.flags,
                )
                continue
            out.append(line)
            continue

        # Other invalid (tiny / low ink): keep for review path with flag
        meta["htr_blocked"] = "destructive_upscale" in vr.flags or "too_tiny" in vr.flags
        meta["htr_block_reason"] = vr.reason
        line.metadata = meta
        out.append(line)

    return out
