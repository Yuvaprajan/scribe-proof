"""Create placeholder held-out image + dirs. Replace with real hard HW pages ASAP."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    held = root / "evaluation" / "dataset" / "held_out"
    img_dir = held / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    (root / "evaluation" / "dataset" / "dev" / "images").mkdir(parents=True, exist_ok=True)
    (root / "evaluation" / "dataset" / "validation" / "images").mkdir(parents=True, exist_ok=True)
    (root / "evaluation" / "reports").mkdir(parents=True, exist_ok=True)

    out = img_dir / "placeholder_messy_note.png"
    h, w = 900, 1200
    img = np.ones((h, w, 3), dtype=np.uint8) * 245
    noise = np.random.normal(0, 6, (h, w, 3))
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    lines = [
        (80, 120, "Patient notes 12 / 03 / 2026"),
        (90, 200, "Dose 500mg twice daily"),
        (100, 280, "Follow up with Dr. Rao"),
        (110, 360, "BP 120 / 80 stable"),
        (700, 180, "margin: urgent"),
    ]
    for x, y, text in lines:
        for dx, dy in [(0, 0), (1, 0), (0, 1)]:
            cv2.putText(
                img,
                text,
                (x + dx, y + dy),
                cv2.FONT_HERSHEY_SCRIPT_COMPLEX if "margin" not in text else cv2.FONT_HERSHEY_SIMPLEX,
                1.1 if "margin" not in text else 0.7,
                (20, 25, 35),
                2,
                cv2.LINE_AA,
            )
    # Simulate a page-like ruled form line (should NOT be crossed-out)
    cv2.line(img, (60, 500), (1100, 500), (180, 180, 180), 2)
    cv2.imwrite(str(out), img)
    print("Wrote", out)
    print("WARNING: placeholder only — add real hard handwriting to held_out before claiming qualification.")


if __name__ == "__main__":
    main()
