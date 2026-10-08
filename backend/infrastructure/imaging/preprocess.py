"""Page normalization and quality profiling (Phase 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.filters import threshold_sauvola


@dataclass
class PageQuality:
    blur_score: float
    contrast_score: float
    skew_angle: float
    resolution: float
    noise_score: float
    quality_class: str
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class VariantBundle:
    raw: np.ndarray
    normalized: np.ndarray
    deskewed: np.ndarray
    clahe: np.ndarray
    adaptive_binarized: np.ndarray
    denoised: np.ndarray
    quality: PageQuality
    settings: dict[str, Any]


class ImagePipeline:
    """Immutable preprocessing transforms — never overwrites source."""

    def __init__(
        self,
        clahe_clip: float = 2.0,
        sauvola_window: int = 25,
        blur_sharpen_threshold: float = 80.0,
    ):
        self.clahe_clip = clahe_clip
        self.sauvola_window = sauvola_window
        self.blur_sharpen_threshold = blur_sharpen_threshold

    def load_from_bytes(self, data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes")
        return img

    def load_with_exif(self, data: bytes) -> np.ndarray:
        from io import BytesIO

        pil = Image.open(BytesIO(data))
        pil = ImageOps.exif_transpose(pil)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        rgb = np.array(pil)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def encode_png(self, image: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Failed to encode PNG")
        return buf.tobytes()

    def encode_jpeg(self, image: np.ndarray, quality: int = 92) -> bytes:
        ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise ValueError("Failed to encode JPEG")
        return buf.tobytes()

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def estimate_blur(self, gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def estimate_contrast(self, gray: np.ndarray) -> float:
        return float(gray.std())

    def estimate_noise(self, gray: np.ndarray) -> float:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = gray.astype(np.float32) - blur.astype(np.float32)
        return float(residual.std())

    def estimate_skew(self, gray: np.ndarray) -> float:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
        if lines is None:
            return 0.0
        angles = []
        for rho_theta in lines[:40]:
            rho, theta = rho_theta[0]
            angle = (theta * 180.0 / np.pi) - 90.0
            if -45 <= angle <= 45:
                angles.append(angle)
        if not angles:
            return 0.0
        return float(np.median(angles))

    def detect_page_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        gray = self._to_gray(image)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape[:2]
        area_img = h * w
        best = None
        best_area = 0
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > best_area and area > 0.3 * area_img:
                    best = approx
                    best_area = area
        return best

    def perspective_correct(self, image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        pts = contour.reshape(4, 2).astype(np.float32)
        # order: tl, tr, br, bl
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        ordered = np.array([tl, tr, br, bl], dtype=np.float32)
        w1 = np.linalg.norm(br - bl)
        w2 = np.linalg.norm(tr - tl)
        h1 = np.linalg.norm(tr - br)
        h2 = np.linalg.norm(tl - bl)
        width = int(max(w1, w2))
        height = int(max(h1, h2))
        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(image, M, (width, height))

    def deskew(self, image: np.ndarray, angle: float) -> np.ndarray:
        if abs(angle) < 0.3:
            return image.copy()
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
            return clahe.apply(image)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

    def sauvola_binarize(self, image: np.ndarray) -> np.ndarray:
        gray = self._to_gray(image)
        window = self.sauvola_window if self.sauvola_window % 2 == 1 else self.sauvola_window + 1
        thresh = threshold_sauvola(gray, window_size=window)
        binary = (gray > thresh).astype(np.uint8) * 255
        return binary

    def denoise(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        return cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)

    def mild_sharpen(self, image: np.ndarray) -> np.ndarray:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)

    def classify_quality(
        self, blur: float, contrast: float, noise: float, resolution: float
    ) -> str:
        score = 0
        if blur >= 120:
            score += 2
        elif blur >= 60:
            score += 1
        if contrast >= 45:
            score += 2
        elif contrast >= 25:
            score += 1
        if noise <= 12:
            score += 2
        elif noise <= 22:
            score += 1
        if resolution >= 150:
            score += 1
        if score >= 6:
            return "good"
        if score >= 4:
            return "fair"
        if score >= 2:
            return "poor"
        return "unusable"

    def process_page(self, raw_bgr: np.ndarray, dpi_estimate: float = 150.0) -> VariantBundle:
        settings: dict[str, Any] = {
            "clahe_clip": self.clahe_clip,
            "sauvola_window": self.sauvola_window,
            "blur_sharpen_threshold": self.blur_sharpen_threshold,
            "dpi_estimate": dpi_estimate,
        }
        raw = raw_bgr.copy()
        normalized = raw.copy()
        contour = self.detect_page_contour(normalized)
        if contour is not None:
            normalized = self.perspective_correct(normalized, contour)
            settings["perspective_correction"] = True
        else:
            settings["perspective_correction"] = False

        gray = self._to_gray(normalized)
        skew = self.estimate_skew(gray)
        deskewed = self.deskew(normalized, skew)
        settings["skew_angle"] = skew

        gray_d = self._to_gray(deskewed)
        blur = self.estimate_blur(gray_d)
        contrast = self.estimate_contrast(gray_d)
        noise = self.estimate_noise(gray_d)
        h, w = gray_d.shape[:2]
        resolution = dpi_estimate

        clahe = self.apply_clahe(deskewed)
        adaptive = self.sauvola_binarize(deskewed)
        # Keep 3-channel for consistent downstream handling where needed
        adaptive_bgr = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
        denoised = self.denoise(deskewed)
        if blur < self.blur_sharpen_threshold:
            denoised = self.mild_sharpen(denoised)
            settings["sharpen_applied"] = True
        else:
            settings["sharpen_applied"] = False

        qclass = self.classify_quality(blur, contrast, noise, resolution)
        quality = PageQuality(
            blur_score=blur,
            contrast_score=contrast,
            skew_angle=skew,
            resolution=resolution,
            noise_score=noise,
            quality_class=qclass,
            settings=settings,
        )
        return VariantBundle(
            raw=raw,
            normalized=normalized,
            deskewed=deskewed,
            clahe=clahe,
            adaptive_binarized=adaptive_bgr,
            denoised=denoised,
            quality=quality,
            settings=settings,
        )

    def crop_bbox(
        self, image: np.ndarray, x: float, y: float, w: float, h: float, pad: int = 4
    ) -> np.ndarray:
        ih, iw = image.shape[:2]
        x0 = max(0, int(x) - pad)
        y0 = max(0, int(y) - pad)
        x1 = min(iw, int(x + w) + pad)
        y1 = min(ih, int(y + h) + pad)
        return image[y0:y1, x0:x1].copy()
