"""Optional VLM verifier — disabled by default (Phase 6)."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

VERIFIER_INSTRUCTION = """You are a visual evidence verifier.
Read only what is visible in the image.
Choose one supplied OCR candidate or return ILLEGIBLE.
Do not create a new transcription.
Do not infer missing words from context.
Return JSON only."""


@dataclass
class VerificationResult:
    selected_candidate_index: Optional[int]
    status: str  # supported | illegible | unavailable | disagreement
    word_confidences: list[float] = field(default_factory=list)
    reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    enabled: bool = False


class VerifierProvider(ABC):
    @abstractmethod
    def verify(
        self,
        image_crop: np.ndarray,
        hypotheses: list[str],
    ) -> VerificationResult:
        ...


class DisabledVerifier(VerifierProvider):
    def verify(
        self,
        image_crop: np.ndarray,
        hypotheses: list[str],
    ) -> VerificationResult:
        return VerificationResult(
            selected_candidate_index=None,
            status="unavailable",
            reason="Verifier disabled (no provider/API key configured).",
            enabled=False,
        )


class HttpVlmVerifier(VerifierProvider):
    """HTTP JSON verifier. Only selects an existing candidate or ILLEGIBLE."""

    def __init__(self, endpoint: str, api_key: str, model: str = "gpt-4o-mini"):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model

    def verify(
        self,
        image_crop: np.ndarray,
        hypotheses: list[str],
    ) -> VerificationResult:
        if not hypotheses:
            return VerificationResult(
                selected_candidate_index=None,
                status="illegible",
                reason="No candidates to verify.",
                enabled=True,
            )
        try:
            import base64
            import cv2
            import httpx

            ok, buf = cv2.imencode(".png", image_crop)
            if not ok:
                raise ValueError("encode failed")
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": VERIFIER_INSTRUCTION},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Candidates:\n"
                                    + "\n".join(f"{i}: {h}" for i, h in enumerate(hypotheses))
                                    + '\nReturn JSON: {"selected_candidate_index":0|null,'
                                    '"status":"supported"|"illegible","word_confidences":[],'
                                    '"reason":"..."}'
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    },
                ],
                "response_format": {"type": "json_object"},
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(self.endpoint, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            content = data["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            idx = parsed.get("selected_candidate_index")
            status = parsed.get("status", "supported")
            # Hard guard: never allow free-form rewrite
            if status == "illegible" or idx is None:
                return VerificationResult(
                    selected_candidate_index=None,
                    status="illegible",
                    word_confidences=parsed.get("word_confidences") or [],
                    reason=parsed.get("reason", "Verifier selected ILLEGIBLE."),
                    raw=parsed,
                    enabled=True,
                )
            idx = int(idx)
            if idx < 0 or idx >= len(hypotheses):
                return VerificationResult(
                    selected_candidate_index=None,
                    status="disagreement",
                    reason="Verifier returned out-of-range index; ignored.",
                    raw=parsed,
                    enabled=True,
                )
            # Reject free-form hallucination if model invents a transcription
            invented = parsed.get("text") or parsed.get("transcription")
            if invented is not None:
                invented_n = " ".join(str(invented).strip().lower().split())
                pool = {" ".join(h.strip().lower().split()) for h in hypotheses}
                if invented_n and invented_n not in pool:
                    return VerificationResult(
                        selected_candidate_index=None,
                        status="disagreement",
                        reason="Verifier invented unsupported transcription; ignored.",
                        raw=parsed,
                        enabled=True,
                    )
            return VerificationResult(
                selected_candidate_index=idx,
                status="supported",
                word_confidences=parsed.get("word_confidences") or [],
                reason=parsed.get("reason", "Candidate supported by verifier."),
                raw=parsed,
                enabled=True,
            )
        except Exception as e:
            logger.warning("Verifier call failed: %s", e)
            return VerificationResult(
                selected_candidate_index=None,
                status="unavailable",
                reason=f"Verifier error: {e}",
                enabled=True,
            )


def build_verifier_from_env() -> VerifierProvider:
    provider = os.getenv("SCRIBEPROOF_VERIFIER_PROVIDER", "auto").strip().lower()
    enabled = os.getenv("SCRIBEPROOF_VERIFIER_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    # Prefer Qwen when provider=qwen or when Qwen is enabled and verifier wants auto
    qwen_key = os.getenv("SCRIBEPROOF_QWEN_API_KEY", "").strip()
    qwen_enabled = os.getenv("SCRIBEPROOF_QWEN_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    qwen_endpoint = os.getenv(
        "SCRIBEPROOF_QWEN_ENDPOINT",
        "https://openrouter.ai/api/v1/chat/completions",
    ).strip()
    qwen_model = os.getenv("SCRIBEPROOF_QWEN_MODEL", "qwen/qwen3-vl-8b-instruct").strip()

    key = os.getenv("SCRIBEPROOF_VERIFIER_API_KEY", "").strip()
    endpoint = os.getenv(
        "SCRIBEPROOF_VERIFIER_ENDPOINT",
        "https://api.openai.com/v1/chat/completions",
    ).strip()
    model = os.getenv("SCRIBEPROOF_VERIFIER_MODEL", "gpt-4o-mini").strip()

    if not enabled:
        return DisabledVerifier()

    if provider == "qwen" or (provider == "auto" and qwen_enabled and qwen_key):
        if qwen_key:
            return HttpVlmVerifier(
                endpoint=qwen_endpoint, api_key=qwen_key, model=qwen_model
            )
    if key:
        return HttpVlmVerifier(endpoint=endpoint, api_key=key, model=model)
    if qwen_key:
        return HttpVlmVerifier(
            endpoint=qwen_endpoint, api_key=qwen_key, model=qwen_model
        )
    return DisabledVerifier()
