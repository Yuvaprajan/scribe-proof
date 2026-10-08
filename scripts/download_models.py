"""Download PaddleOCR + TrOCR weights for offline/local runs."""

from __future__ import annotations

import os
import sys

# Torch before paddle on Windows
import torch  # noqa: F401

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "hf_cache"))


def main() -> None:
    model = os.getenv(
        "SCRIBEPROOF_TROCR_MODEL", "microsoft/trocr-large-handwritten"
    )
    print("Downloading TrOCR:", model)
    from huggingface_hub import snapshot_download

    path = snapshot_download(model, max_workers=2)
    print("TrOCR cached at", path)

    print("Warming PaddleOCR detectors…")
    from infrastructure.models.detection import LineDetector

    det = LineDetector(use_paddle=True)
    ok = det.warmup()
    print("Paddle ready:" if ok else "Paddle failed:", det.status)

    print("Warming TrOCR…")
    from infrastructure.models.recognition import TrOCRRecognitionProvider

    rec = TrOCRRecognitionProvider(model_name=model)
    ok = rec.warmup()
    print("TrOCR ready:" if ok else "TrOCR failed:", rec.status)


if __name__ == "__main__":
    main()
