"""P0 tests: detection guards, TrOCR input contract, splits."""

from __future__ import annotations

import numpy as np
import pytest

from domain.entities import BoundingBox
from infrastructure.models.bbox_validation import (
    BBoxThresholds,
    estimate_line_count,
    split_region_horizontal,
    validate_bbox,
    validate_trocr_crop,
)
from infrastructure.models.recognition import prepare_trocr_image


def _blank(h=800, w=600, color=255):
    return np.full((h, w, 3), color, dtype=np.uint8)


def _ink_line(img, y=40, text_rows=18):
    """Draw a dark horizontal ink band."""
    img[y : y + text_rows, 20:-20] = 30
    return img


def test_page_sized_bbox_rejected():
    page = _blank(800, 600)
    bbox = BoundingBox(x=10, y=10, width=580, height=780)
    crop = page[10:790, 10:590]
    vr = validate_bbox(bbox, 600, 800, crop)
    assert vr.reject_for_htr or vr.needs_split
    assert "excessive_area" in vr.flags or "excessive_height" in vr.flags


def test_valid_line_accepted():
    page = _blank(800, 600)
    _ink_line(page, y=100, text_rows=28)
    bbox = BoundingBox(x=20, y=95, width=400, height=40)
    crop = page[95:135, 20:420]
    vr = validate_bbox(bbox, 600, 800, crop)
    assert vr.ok
    assert not vr.reject_for_htr


def test_trocr_rejects_page_crop():
    page = _blank(400, 400)
    page[20:380, 20:380] = 40
    with pytest.raises(ValueError, match="trocr_input_rejected"):
        prepare_trocr_image(page)


def test_trocr_accepts_line_crop():
    line = _blank(40, 300)
    _ink_line(line, y=8, text_rows=20)
    out = prepare_trocr_image(line)
    assert out.shape[0] > 0
    assert out.shape[2] == 3


def test_tiny_crop_destructive_upscale_flagged():
    tiny = np.full((4, 40, 3), 40, dtype=np.uint8)
    vr = validate_trocr_crop(tiny)
    assert "destructive_upscale" in vr.flags or "too_tiny" in vr.flags
    assert vr.reject_for_htr


def test_projection_split_multi_line():
    page = _blank(200, 400)
    _ink_line(page, y=20, text_rows=16)
    _ink_line(page, y=80, text_rows=16)
    _ink_line(page, y=140, text_rows=16)
    bbox = BoundingBox(x=0, y=0, width=400, height=200)
    children = split_region_horizontal(page, bbox)
    assert len(children) >= 2


def test_estimate_line_count():
    page = _blank(150, 300)
    _ink_line(page, y=10, text_rows=14)
    _ink_line(page, y=70, text_rows=14)
    n = estimate_line_count(page)
    assert n >= 2
