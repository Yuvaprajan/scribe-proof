"""Crossed-out vs ruling/form lines."""

from __future__ import annotations

import numpy as np
import cv2

from infrastructure.models.detection import LineDetector


def test_ruling_line_not_crossed_out():
    img = np.full((80, 300, 3), 245, dtype=np.uint8)
    # faint text band (upper)
    img[18:32, 30:270] = 50
    # form ruling at bottom edge (should be underline/ruling/border, not strike)
    cv2.line(img, (5, 74), (295, 74), (90, 90, 90), 2)
    det = LineDetector(use_paddle=False)
    crossed, _ = det._crossed_out_score(img)
    # Product rule: form rulings must not become CROSSED_OUT
    assert crossed is False


def test_empty_crop_not_crossed_out():
    det = LineDetector(use_paddle=False)
    assert det._crossed_out_score(np.zeros((0, 0, 3), dtype=np.uint8)) == (False, 0.0)


def test_border_line_not_crossed_out():
    img = np.full((100, 400, 3), 250, dtype=np.uint8)
    img[40:55, 40:360] = 60
    cv2.rectangle(img, (5, 5), (395, 95), (80, 80, 80), 2)
    det = LineDetector(use_paddle=False)
    crossed, _ = det._crossed_out_score(img)
    assert crossed is False
