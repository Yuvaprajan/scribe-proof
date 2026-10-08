"""Generate a messy synthetic handwriting sample for local demo."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "samples" / "messy_note.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    h, w = 900, 1200
    img = np.ones((h, w, 3), dtype=np.uint8) * 245
    # paper texture
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
        # Simulate shaky ink strokes via repeated slight offsets
        for dx, dy in [(0, 0), (1, 0), (0, 1), (-1, 0)]:
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

    # Crossed-out line
    y = 440
    cv2.putText(
        img,
        "Old dose 250mg cancelled",
        (100, y),
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        1.0,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    cv2.line(img, (95, y - 12), (620, y - 8), (15, 15, 15), 3)

    # Mild skew + blur
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 1.8, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=(245, 245, 245))
    img = cv2.GaussianBlur(img, (3, 3), 0)

    cv2.imwrite(str(out), img)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
