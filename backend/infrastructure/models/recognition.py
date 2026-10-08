"""Handwriting recognition — TrOCR primary; Paddle print-OCR never overrides."""

from __future__ import annotations

import logging
import math
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Optional

import cv2
import numpy as np

from domain.entities import Hypothesis, new_id
from domain.policy import normalize_candidate

logger = logging.getLogger(__name__)

TROCR_BEAM_CONFIG = {
    "num_beams": 8,
    "num_return_sequences": 5,
    "output_scores": True,
    "return_dict_in_generate": True,
    "early_stopping": True,
}

# Fast path for CPU demos / first pass (escalate uses full beam config)
TROCR_FAST_CONFIG = {
    "num_beams": 4,
    "num_return_sequences": 2,
    "output_scores": True,
    "return_dict_in_generate": True,
    "early_stopping": True,
}


class RecognitionProvider(ABC):
    MODEL_NAME: str = "unknown"
    MODEL_VERSION: str = "unknown"

    @abstractmethod
    def recognize(
        self,
        image: np.ndarray,
        options: Optional[dict[str, Any]] = None,
    ) -> list[Hypothesis]:
        ...

    def warmup(self) -> bool:
        return True

    @property
    def status(self) -> dict[str, Any]:
        return {
            "provider": type(self).__name__,
            "model_name": self.MODEL_NAME,
            "model_version": self.MODEL_VERSION,
            "ready": True,
        }


def prepare_trocr_image(image: np.ndarray, *, skip_validation: bool = False) -> np.ndarray:
    """Handwriting-oriented crop prep for TrOCR.

    Must only be called on validated line crops. Page-sized / multi-line
    crops are rejected by validate_trocr_crop unless skip_validation=True
    (tests only).
    """
    if image is None or image.size == 0:
        return np.full((64, 256, 3), 255, dtype=np.uint8)

    if not skip_validation:
        from infrastructure.models.bbox_validation import validate_trocr_crop

        vr = validate_trocr_crop(image)
        if vr.reject_for_htr or not vr.ok:
            raise ValueError(f"trocr_input_rejected:{vr.reason}")

    if len(image.shape) == 2:
        gray = image
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mild deskew on the crop itself
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=40)
    angle = 0.0
    if lines is not None:
        angles = []
        for rho_theta in lines[:20]:
            _, theta = rho_theta[0]
            a = (theta * 180.0 / np.pi) - 90.0
            if -20 <= a <= 20:
                angles.append(a)
        if angles:
            angle = float(np.median(angles))
    if abs(angle) >= 0.5:
        h0, w0 = bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), angle, 1.0)
        bgr = cv2.warpAffine(
            bgr, M, (w0, h0), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Contrast for faint ink
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Target line height ~64–96px (TrOCR likes readable stroke scale)
    h, w = gray.shape[:2]
    target_h = 80
    if h < 1:
        return np.full((64, 256, 3), 255, dtype=np.uint8)
    scale = target_h / float(h)
    # Avoid extreme upscaling artifacts
    scale = min(max(scale, 0.5), 4.0)
    new_w = max(32, int(w * scale))
    gray = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    pad_x = max(12, new_w // 16)
    pad_y = 10
    gray = cv2.copyMakeBorder(
        gray, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=255
    )
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return rgb


_GARBAGE_RE = re.compile(r"^[^A-Za-z0-9]+$")


def is_garbage_transcription(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t) == 1 and not t.isalnum():
        return True
    if _GARBAGE_RE.match(t):
        return True
    # Extreme repetition often from failed decode
    if len(set(t.replace(" ", ""))) <= 1 and len(t) > 3:
        return True
    return False


def calibrate_trocr_visual(seq_score: float, rank: int) -> float:
    """Map TrOCR sequence log-prob to a conservative visual score."""
    # sequences_scores are typically negative; higher (closer to 0) is better
    # Use a steeper sigmoid so weak beams don't look "confident"
    visual = 1.0 / (1.0 + math.exp(-(seq_score + 1.5) / 1.5))
    visual *= max(0.55, 1.0 - 0.08 * (rank - 1))
    return float(max(0.02, min(0.95, visual)))


class TrOCRRecognitionProvider(RecognitionProvider):
    """microsoft/trocr-*-handwritten via Hugging Face Transformers + beam search."""

    def __init__(self, device: Optional[str] = None, model_name: Optional[str] = None):
        self.MODEL_NAME = model_name or os.getenv(
            "SCRIBEPROOF_TROCR_MODEL", "microsoft/trocr-base-handwritten"
        )
        self.MODEL_VERSION = "hf-transformers"
        self.device = device
        self._processor = None
        self._model = None
        self._load_error: Optional[str] = None
        self._loaded = False
        self._dtype = None  # torch.dtype when on CUDA

    @property
    def status(self) -> dict[str, Any]:
        return {
            "provider": "TrOCRRecognitionProvider",
            "model_name": self.MODEL_NAME,
            "model_version": self.MODEL_VERSION,
            "ready": self._loaded,
            "device": self.device,
            "dtype": str(self._dtype) if self._dtype is not None else None,
            "error": self._load_error,
        }

    def warmup(self) -> bool:
        return self._ensure_loaded()

    def _ensure_loaded(self) -> bool:
        if self._loaded and self._model is not None:
            return True
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            logger.info("Loading TrOCR model %s …", self.MODEL_NAME)
            self._processor = TrOCRProcessor.from_pretrained(self.MODEL_NAME)
            try:
                self._model = VisionEncoderDecoderModel.from_pretrained(
                    self.MODEL_NAME, use_safetensors=True
                )
            except Exception:
                self._model = VisionEncoderDecoderModel.from_pretrained(self.MODEL_NAME)
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # 4GB GPUs (e.g. RTX 2050): fp16 is required for trocr-large
            dtype_env = os.getenv("SCRIBEPROOF_TROCR_DTYPE", "auto").lower()
            self._dtype = torch.float32
            if self.device.startswith("cuda"):
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                want_half = dtype_env in {"fp16", "float16", "half"} or (
                    dtype_env == "auto"
                    and (vram_gb < 6.5 or "large" in self.MODEL_NAME.lower())
                )
                if want_half:
                    self._dtype = torch.float16
                    self._model = self._model.to(device=self.device, dtype=self._dtype)
                else:
                    self._model = self._model.to(self.device)
            else:
                self._model = self._model.to(self.device)

            self._model.eval()
            self._model.config.decoder.is_decoder = True
            self._loaded = True
            self._load_error = None
            logger.info(
                "TrOCR loaded on %s dtype=%s", self.device, self._dtype
            )
            return True
        except Exception as e:
            self._load_error = str(e)
            self._loaded = False
            self._model = None
            self._processor = None
            logger.error("Failed to load TrOCR: %s", e)
            return False

    def recognize(
        self,
        image: np.ndarray,
        options: Optional[dict[str, Any]] = None,
    ) -> list[Hypothesis]:
        options = options or {}
        run_id = options.get("recognition_run_id", new_id())
        source_crop_id = options.get("source_crop_id", "")
        image_variant = options.get("image_variant", "raw")
        fast = bool(options.get("fast", False))

        if not self._ensure_loaded():
            return [
                Hypothesis(
                    recognition_run_id=run_id,
                    text="",
                    normalized_text="",
                    rank=1,
                    sequence_score=-100.0,
                    visual_score=0.0,
                    model_name=self.MODEL_NAME,
                    model_version=self.MODEL_VERSION,
                    source_crop_id=source_crop_id,
                    image_variant=image_variant,
                    configuration={"error": self._load_error, **TROCR_BEAM_CONFIG},
                )
            ]

        import torch
        from PIL import Image

        try:
            rgb = prepare_trocr_image(image)
        except ValueError as e:
            logger.warning("TrOCR skipped invalid crop: %s", e)
            return [
                Hypothesis(
                    recognition_run_id=run_id,
                    text="",
                    normalized_text="",
                    rank=1,
                    sequence_score=-100.0,
                    visual_score=0.0,
                    model_name=self.MODEL_NAME,
                    model_version=self.MODEL_VERSION,
                    source_crop_id=source_crop_id,
                    image_variant=image_variant,
                    configuration={
                        "error": str(e),
                        "family": "trocr",
                        "htr_blocked": True,
                        **TROCR_BEAM_CONFIG,
                    },
                )
            ]

        pil = Image.fromarray(rgb.astype(np.uint8))
        pixel_values = self._processor(images=pil, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(
            device=self.device,
            dtype=self._dtype if self._dtype is not None else torch.float32,
        )

        cfg = TROCR_FAST_CONFIG if fast else TROCR_BEAM_CONFIG
        gen_kwargs = {
            "num_beams": int(options.get("num_beams", cfg["num_beams"])),
            "num_return_sequences": int(
                options.get("num_return_sequences", cfg["num_return_sequences"])
            ),
            "output_scores": True,
            "return_dict_in_generate": True,
            "early_stopping": True,
            "max_new_tokens": int(options.get("max_new_tokens", 48)),
        }

        # inference_mode is faster than no_grad on CPU/GPU for decode loops
        with torch.inference_mode():
            outputs = self._model.generate(pixel_values, **gen_kwargs)
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        sequences = outputs.sequences
        scores = getattr(outputs, "sequences_scores", None)
        decoded = self._processor.batch_decode(sequences, skip_special_tokens=True)

        hyps: list[Hypothesis] = []
        for i, text in enumerate(decoded):
            text = text.strip()
            if is_garbage_transcription(text):
                text = ""
            seq_score = float(scores[i].item()) if scores is not None else -float(i + 1)
            visual = (
                calibrate_trocr_visual(seq_score, i + 1)
                if scores is not None
                else max(0.05, 0.65 - 0.12 * i)
            )
            if not text:
                visual = min(visual, 0.08)
            hyps.append(
                Hypothesis(
                    recognition_run_id=run_id,
                    text=text,
                    normalized_text=normalize_candidate(text),
                    rank=i + 1,
                    sequence_score=seq_score,
                    visual_score=float(visual),
                    model_name=self.MODEL_NAME,
                    model_version=self.MODEL_VERSION,
                    source_crop_id=source_crop_id,
                    image_variant=image_variant,
                    configuration={**gen_kwargs, "family": "trocr"},
                )
            )
        return hyps


class EnsembleRecognitionProvider(RecognitionProvider):
    """Adaptive ensemble: TrOCR-primary for handwriting; Paddle-primary for print."""

    MODEL_NAME = "trocr+paddle-ensemble"
    MODEL_VERSION = "3.0"

    # Handwriting: Paddle must never outrank TrOCR.
    PADDLE_VISUAL_CAP = 0.32

    def __init__(self, trocr: TrOCRRecognitionProvider):
        self.trocr = trocr
        self.MODEL_NAME = trocr.MODEL_NAME
        self._trocr_ready = False

    def warmup(self) -> bool:
        self._trocr_ready = self.trocr.warmup()
        return self._trocr_ready

    @property
    def status(self) -> dict[str, Any]:
        st = self.trocr.status
        st["provider"] = "EnsembleRecognitionProvider"
        st["trocr_ready"] = self._trocr_ready or bool(st.get("ready"))
        st["paddle_hypotheses"] = "adaptive_print_or_weak"
        st["policy"] = "adaptive_print_handwriting"
        return st

    def recognize(
        self,
        image: np.ndarray,
        options: Optional[dict[str, Any]] = None,
    ) -> list[Hypothesis]:
        options = options or {}
        run_id = options.get("recognition_run_id", new_id())
        source_crop_id = options.get("source_crop_id", "")
        image_variant = options.get("image_variant", "raw")
        paddle_text = (options.get("paddle_text") or "").strip()
        paddle_score = float(options.get("paddle_score") or 0.0)
        script_mode = (options.get("script_mode") or "handwriting").lower()
        region_type = (options.get("region_type") or "").lower()
        print_mode = script_mode == "print" or region_type in {
            "printed_text",
            "table",
        }

        hyps: list[Hypothesis] = []

        # Printed text: trust Paddle first (skip expensive TrOCR when strong).
        if print_mode and paddle_text and not is_garbage_transcription(paddle_text):
            visual = float(max(0.55, min(0.95, paddle_score)))
            hyps.append(
                Hypothesis(
                    recognition_run_id=run_id,
                    text=paddle_text,
                    normalized_text=normalize_candidate(paddle_text),
                    rank=1,
                    sequence_score=math.log(max(visual, 1e-3)),
                    visual_score=visual,
                    model_name="paddleocr-rec",
                    model_version="paddle",
                    source_crop_id=source_crop_id,
                    image_variant=image_variant,
                    configuration={
                        "source": "paddle_print_ocr_primary",
                        "trust": "high",
                        "script_mode": "print",
                        "family": "paddle",
                    },
                )
            )
            # Optional weak TrOCR only when Paddle is middling
            if paddle_score < 0.85 and self.trocr._ensure_loaded():
                self._trocr_ready = True
                for h in self.trocr.recognize(image, {**options, "fast": True}):
                    if h.configuration.get("error") or not h.text:
                        continue
                    h.visual_score = min(h.visual_score, 0.45)
                    h.configuration = {**(h.configuration or {}), "role": "print_secondary"}
                    hyps.append(h)
        else:
            if self.trocr._ensure_loaded():
                self._trocr_ready = True
                hyps = [
                    h
                    for h in self.trocr.recognize(image, options)
                    if not h.configuration.get("error")
                ]
            else:
                self._trocr_ready = False

            # Handwriting: Paddle is weak disagreement evidence only
            if paddle_text and not is_garbage_transcription(paddle_text):
                existing = {normalize_candidate(h.text) for h in hyps if h.text}
                norm = normalize_candidate(paddle_text)
                if norm and norm not in existing:
                    weak = min(self.PADDLE_VISUAL_CAP, max(0.05, paddle_score * 0.35))
                    hyps.append(
                        Hypothesis(
                            recognition_run_id=run_id,
                            text=paddle_text,
                            normalized_text=norm,
                            rank=len(hyps) + 1,
                            sequence_score=math.log(max(weak, 1e-3)),
                            visual_score=float(weak),
                            model_name="paddleocr-rec",
                            model_version="paddle",
                            source_crop_id=source_crop_id,
                            image_variant=image_variant,
                            configuration={
                                "source": "paddle_print_ocr_weak",
                                "trust": "low",
                                "script_mode": "handwriting",
                                "note": "Print OCR hint only — not authoritative for handwriting",
                            },
                        )
                    )

        if not hyps:
            hyps.append(
                Hypothesis(
                    recognition_run_id=run_id,
                    text="",
                    normalized_text="",
                    rank=1,
                    sequence_score=-100.0,
                    visual_score=0.0,
                    model_name=self.MODEL_NAME,
                    model_version=self.MODEL_VERSION,
                    source_crop_id=source_crop_id,
                    image_variant=image_variant,
                    configuration={"error": "no_usable_hypotheses"},
                )
            )

        hyps.sort(key=lambda h: (-h.visual_score, h.rank))
        for i, h in enumerate(hyps):
            h.rank = i + 1
        return hyps


class MockRecognitionProvider(RecognitionProvider):
    MODEL_NAME = "mock-trocr"
    MODEL_VERSION = "demo-1"

    def recognize(
        self,
        image: np.ndarray,
        options: Optional[dict[str, Any]] = None,
    ) -> list[Hypothesis]:
        options = options or {}
        run_id = options.get("recognition_run_id", new_id())
        source_crop_id = options.get("source_crop_id", "")
        image_variant = options.get("image_variant", "raw")
        return [
            Hypothesis(
                recognition_run_id=run_id,
                text="",
                normalized_text="",
                rank=1,
                sequence_score=-100.0,
                visual_score=0.0,
                model_name=self.MODEL_NAME,
                model_version=self.MODEL_VERSION,
                source_crop_id=source_crop_id,
                image_variant=image_variant,
                configuration={"mode": "mock"},
            )
        ]
