"""Qwen3-VL document understanding — category, layout, structured line items."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from domain.entities import DocumentCategory, RegionType, new_id

logger = logging.getLogger(__name__)

UNDERSTAND_INSTRUCTION = """You are a document layout and content analyst.
Analyze the image carefully. Return JSON only (no markdown).
Do not invent text that is not visible. If unsure, use empty string or null.

Schema:
{
  "category": one of [
    "prescription","medical_report","lab_report","form","letter",
    "architecture_diagram","flowchart","sketch","receipt","id_card",
    "handwritten_note","mixed","other","unknown"
  ],
  "category_confidence": 0.0-1.0,
  "layout_regions": [
    {"type": "printed_text|main_handwriting|margin_note|table|signature|diagram|sketch|unknown",
     "description": "short description",
     "bbox_norm": [x0,y0,x1,y1]  // normalized 0-1 page coords, optional}
  ],
  "line_items": [
    {"label": "field or element name",
     "value": "visible value or description",
     "kind": "field|heading|paragraph|table_row|diagram_element|note|other"}
  ],
  "summary": "one sentence describing what this document is"
}
"""

VALID_CATEGORIES = {c.value for c in DocumentCategory}

TYPE_MAP = {
    "printed_text": RegionType.PRINTED_TEXT,
    "print": RegionType.PRINTED_TEXT,
    "printed": RegionType.PRINTED_TEXT,
    "main_handwriting": RegionType.MAIN_HANDWRITING,
    "handwriting": RegionType.MAIN_HANDWRITING,
    "handwritten": RegionType.MAIN_HANDWRITING,
    "margin_note": RegionType.MARGIN_NOTE,
    "margin": RegionType.MARGIN_NOTE,
    "table": RegionType.TABLE,
    "signature": RegionType.SIGNATURE,
    "diagram": RegionType.UNKNOWN,
    "sketch": RegionType.UNKNOWN,
    "flowchart": RegionType.UNKNOWN,
    "architecture_diagram": RegionType.UNKNOWN,
    "unknown": RegionType.UNKNOWN,
}


@dataclass
class LayoutRegionHint:
    region_type: RegionType
    description: str = ""
    bbox_norm: Optional[list[float]] = None  # [x0,y0,x1,y1] 0-1
    raw_type: str = ""


@dataclass
class LineItem:
    id: str
    label: str
    value: str
    kind: str = "field"
    source: str = "qwen"
    confidence: float = 0.5


@dataclass
class DocumentUnderstanding:
    category: DocumentCategory = DocumentCategory.UNKNOWN
    category_confidence: float = 0.0
    layout_regions: list[LayoutRegionHint] = field(default_factory=list)
    line_items: list[LineItem] = field(default_factory=list)
    summary: str = ""
    enabled: bool = False
    provider: str = "disabled"
    raw: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def as_metadata(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "category_confidence": self.category_confidence,
            "layout_summary": [
                {
                    "region_type": r.region_type.value,
                    "description": r.description,
                    "bbox_norm": r.bbox_norm,
                    "raw_type": r.raw_type,
                }
                for r in self.layout_regions
            ],
            "line_items": [
                {
                    "id": i.id,
                    "label": i.label,
                    "value": i.value,
                    "kind": i.kind,
                    "source": i.source,
                    "confidence": i.confidence,
                }
                for i in self.line_items
            ],
            "summary": self.summary,
            "understanding_provider": self.provider,
            "understanding_enabled": self.enabled,
            "understanding_error": self.error,
        }


def heuristic_understanding(
    *,
    paddle_print_ratio: float = 0.0,
    line_count: int = 0,
) -> DocumentUnderstanding:
    """Offline fallback when Qwen is disabled/unavailable."""
    if paddle_print_ratio >= 0.6 and line_count > 0:
        cat = DocumentCategory.OTHER
        conf = 0.35
        layout = [
            LayoutRegionHint(
                region_type=RegionType.PRINTED_TEXT,
                description="Likely printed text (heuristic)",
                raw_type="printed_text",
            )
        ]
    elif line_count > 0:
        cat = DocumentCategory.HANDWRITTEN_NOTE
        conf = 0.3
        layout = [
            LayoutRegionHint(
                region_type=RegionType.MAIN_HANDWRITING,
                description="Likely handwriting (heuristic)",
                raw_type="main_handwriting",
            )
        ]
    else:
        cat = DocumentCategory.UNKNOWN
        conf = 0.1
        layout = []
    return DocumentUnderstanding(
        category=cat,
        category_confidence=conf,
        layout_regions=layout,
        line_items=[],
        summary="Heuristic layout estimate (Qwen disabled).",
        enabled=False,
        provider="heuristic",
    )


def _parse_category(raw: str) -> DocumentCategory:
    key = (raw or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "rx": DocumentCategory.PRESCRIPTION,
        "script": DocumentCategory.PRESCRIPTION,
        "diagram": DocumentCategory.ARCHITECTURE_DIAGRAM,
        "architecture": DocumentCategory.ARCHITECTURE_DIAGRAM,
        "flow_chart": DocumentCategory.FLOWCHART,
        "handwriting": DocumentCategory.HANDWRITTEN_NOTE,
        "note": DocumentCategory.HANDWRITTEN_NOTE,
        "report": DocumentCategory.MEDICAL_REPORT,
    }
    if key in VALID_CATEGORIES:
        return DocumentCategory(key)
    if key in aliases:
        return aliases[key]
    return DocumentCategory.UNKNOWN


def _parse_region_type(raw: str) -> RegionType:
    key = (raw or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
    return TYPE_MAP.get(key, RegionType.UNKNOWN)


class DocumentUnderstandingProvider:
    def understand(self, image_bgr: np.ndarray) -> DocumentUnderstanding:
        raise NotImplementedError


class DisabledUnderstanding(DocumentUnderstandingProvider):
    def understand(self, image_bgr: np.ndarray) -> DocumentUnderstanding:
        return heuristic_understanding()


class HttpQwenVLClient(DocumentUnderstandingProvider):
    """OpenAI-compatible chat/completions client for Qwen3-VL."""

    def __init__(self, endpoint: str, api_key: str, model: str):
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/chat/completions"):
            # Allow base like .../v1
            if self.endpoint.endswith("/v1"):
                self.endpoint = self.endpoint + "/chat/completions"
            elif "/chat/completions" not in self.endpoint:
                self.endpoint = self.endpoint + "/chat/completions"
        self.api_key = api_key
        self.model = model

    def understand(self, image_bgr: np.ndarray) -> DocumentUnderstanding:
        try:
            import httpx

            ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                raise ValueError("image encode failed")
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": UNDERSTAND_INSTRUCTION},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Classify this document, describe layout regions, and list visible line items as JSON.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                        ],
                    },
                ],
                "temperature": 0.1,
                "max_tokens": 2048,
                "response_format": {"type": "json_object"},
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            # OpenRouter recommends these; harmless for other providers
            if "openrouter.ai" in self.endpoint:
                headers["HTTP-Referer"] = os.getenv(
                    "SCRIBEPROOF_OPENROUTER_REFERER", "https://scribeproof.local"
                )
                headers["X-Title"] = os.getenv(
                    "SCRIBEPROOF_OPENROUTER_TITLE", "ScribeProof"
                )
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(self.endpoint, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                # Some providers return multimodal content blocks
                texts = [
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and c.get("type") in {None, "text"}
                ]
                content = "\n".join(texts) if texts else str(content)
            parsed = self._extract_json(str(content))
            return self._from_parsed(parsed, provider=f"qwen:{self.model}")
        except Exception as e:
            logger.warning("Qwen understand failed: %s", e)
            out = heuristic_understanding()
            out.error = str(e)
            out.provider = f"qwen_error:{self.model}"
            return out

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                return json.loads(m.group(0))
            raise

    def _from_parsed(self, parsed: dict[str, Any], *, provider: str) -> DocumentUnderstanding:
        cat = _parse_category(str(parsed.get("category", "unknown")))
        conf = float(parsed.get("category_confidence") or 0.5)
        conf = max(0.0, min(1.0, conf))
        regions: list[LayoutRegionHint] = []
        for r in parsed.get("layout_regions") or []:
            if not isinstance(r, dict):
                continue
            raw_type = str(r.get("type") or "unknown")
            bbox = r.get("bbox_norm")
            bbox_norm = None
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    bbox_norm = [float(x) for x in bbox]
                except (TypeError, ValueError):
                    bbox_norm = None
            regions.append(
                LayoutRegionHint(
                    region_type=_parse_region_type(raw_type),
                    description=str(r.get("description") or ""),
                    bbox_norm=bbox_norm,
                    raw_type=raw_type,
                )
            )
        items: list[LineItem] = []
        for it in parsed.get("line_items") or []:
            if not isinstance(it, dict):
                continue
            label = str(it.get("label") or "").strip()
            value = str(it.get("value") or "").strip()
            if not label and not value:
                continue
            items.append(
                LineItem(
                    id=new_id(),
                    label=label or "item",
                    value=value,
                    kind=str(it.get("kind") or "field"),
                    source="qwen",
                    confidence=conf,
                )
            )
        return DocumentUnderstanding(
            category=cat,
            category_confidence=conf,
            layout_regions=regions,
            line_items=items,
            summary=str(parsed.get("summary") or ""),
            enabled=True,
            provider=provider,
            raw=parsed,
        )


def build_understanding_from_env() -> DocumentUnderstandingProvider:
    enabled = os.getenv("SCRIBEPROOF_QWEN_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    key = os.getenv("SCRIBEPROOF_QWEN_API_KEY", "").strip()
    endpoint = os.getenv(
        "SCRIBEPROOF_QWEN_ENDPOINT",
        "https://openrouter.ai/api/v1/chat/completions",
    ).strip()
    model = os.getenv("SCRIBEPROOF_QWEN_MODEL", "qwen/qwen3-vl-8b-instruct").strip()
    if enabled and key:
        return HttpQwenVLClient(endpoint=endpoint, api_key=key, model=model)
    return DisabledUnderstanding()


def apply_layout_hints_to_lines(
    lines: list[Any],
    understanding: DocumentUnderstanding,
    page_w: int,
    page_h: int,
) -> None:
    """Stamp DetectedLine.region_type from Qwen layout bbox hints when they overlap."""
    hints = [h for h in understanding.layout_regions if h.bbox_norm]
    if not hints:
        # Document-level bias: mostly print category → boost printed_text for high paddle scores
        printish = understanding.category in {
            DocumentCategory.FORM,
            DocumentCategory.LETTER,
            DocumentCategory.RECEIPT,
            DocumentCategory.ID_CARD,
            DocumentCategory.LAB_REPORT,
            DocumentCategory.MEDICAL_REPORT,
        }
        for line in lines:
            paddle_score = float(getattr(line, "paddle_score", 0) or 0)
            meta = getattr(line, "metadata", {}) or {}
            if printish and paddle_score >= 0.75 and meta.get("script_mode") != "handwriting":
                if line.region_type in {
                    RegionType.MAIN_HANDWRITING,
                    RegionType.UNKNOWN,
                }:
                    line.region_type = RegionType.PRINTED_TEXT
                    meta["layout_source"] = "qwen_category_bias"
                    line.metadata = meta
        return

    for line in lines:
        bbox = line.bbox
        cx = (bbox.x + bbox.width / 2) / max(page_w, 1)
        cy = (bbox.y + bbox.height / 2) / max(page_h, 1)
        best = None
        best_area = 1e9
        for h in hints:
            x0, y0, x1, y1 = h.bbox_norm  # type: ignore[misc]
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                area = max(1e-6, (x1 - x0) * (y1 - y0))
                if area < best_area:
                    best_area = area
                    best = h
        if best and best.region_type not in {
            RegionType.CROSSED_OUT_CANDIDATE,
        }:
            # Don't override crossed-out
            if line.region_type != RegionType.CROSSED_OUT_CANDIDATE:
                line.region_type = best.region_type
                meta = getattr(line, "metadata", {}) or {}
                meta["layout_source"] = "qwen_bbox"
                meta["layout_description"] = best.description
                line.metadata = meta
