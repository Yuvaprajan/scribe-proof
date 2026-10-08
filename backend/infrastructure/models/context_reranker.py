"""Safe contextual reranking — lexicon/format repairs only; never invents free text."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz, process

from domain.entities import Hypothesis, new_id
from domain.policy import is_trocr_hyp, normalize_candidate


DATE_RE = re.compile(
    r"^(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})$"
)
QTY_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(mg|ml|g|kg|mcg|iu|%|mm|cm)$", re.I)

# Common clinical / note vocabulary for constrained token repair (hackathon default).
# Corrections only fire when a decoded token is already near a lexicon entry.
DEFAULT_DOMAIN_TERMS = [
    "patient",
    "allergy",
    "allergies",
    "nkda",
    "notes",
    "note",
    "follow",
    "fever",
    "persists",
    "amoxicillin",
    "ibuprofen",
    "acetaminophen",
    "metformin",
    "lisinopril",
    "atorvastatin",
    "prescription",
    "rx",
    "dose",
    "dosage",
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "daily",
    "twice",
    "tid",
    "bid",
    "qid",
    "prn",
    "food",
    "with",
    "take",
    "days",
    "weeks",
    "date",
    "doctor",
    "diagnosis",
    "history",
    "blood",
    "pressure",
    "mg",
    "ml",
    "jordan",
    "miles",
]

ABBREV = {
    "dr": "Dr",
    "mr": "Mr",
    "mrs": "Mrs",
    "rx": "Rx",
    "nkda": "NKDA",
    "tid": "tid",
    "bid": "bid",
    "qid": "qid",
    "prn": "prn",
}


@dataclass
class RerankResult:
    hypotheses: list[Hypothesis]
    selected: Optional[Hypothesis]
    format_edit: Optional[dict[str, Any]] = None
    context_score: float = 0.0


class ContextReranker:
    def __init__(self, domain_terms: Optional[list[str]] = None):
        env_terms = [
            t.strip()
            for t in os.getenv("SCRIBEPROOF_DOMAIN_LEXICON", "").split(",")
            if t.strip()
        ]
        terms = domain_terms if domain_terms is not None else DEFAULT_DOMAIN_TERMS
        self.domain_terms = sorted(set(t.lower() for t in list(terms) + env_terms))
        self.units = {"mg", "ml", "g", "kg", "mcg", "iu", "mm", "cm", "%"}
        # Max normalized token edit distance to allow lexicon swap
        self.max_token_dist = float(os.getenv("SCRIBEPROOF_LEXICON_MAX_DIST", "0.34"))

    def _format_only(self, text: str) -> tuple[str, Optional[str]]:
        """Return (possibly reformatted text, edit_type) without inventing content."""
        t = normalize_candidate(text)
        edit: Optional[str] = None
        compact = t.replace(" ", "")
        m = DATE_RE.match(compact)
        if m:
            a, b, c = m.group(1), m.group(2), m.group(3)
            formatted = f"{int(a):02d}/{int(b):02d}/{c}"
            if formatted != t:
                return formatted, "date_format"
        m2 = re.match(r"^(\d+(?:\.\d+)?)(mg|ml|g|kg|mcg|iu|mm|cm)$", t, re.I)
        if m2:
            formatted = f"{m2.group(1)} {m2.group(2).lower()}"
            if formatted != t:
                return formatted, "unit_spacing"
        m3 = re.match(r"^(\d+(?:\.\d+)?)\s+/\s+(\d+(?:\.\d+)?)$", t)
        if m3:
            formatted = f"{m3.group(1)}/{m3.group(2)}"
            if formatted != t:
                return formatted, "slash_normalize"
        # Inline unit OCR slips: 500ma / 500 mq → 500 mg
        repaired_units, n_unit = re.subn(
            r"\b(\d+(?:\.\d+)?)\s*(ma|mq|ng)\b",
            lambda m: f"{m.group(1)} mg",
            t,
            flags=re.I,
        )
        if n_unit:
            t = repaired_units
            edit = "unit_repair"
        # Common prescription-prefix slips near Rx
        repaired_rx, n_rx = re.subn(r"\b(Rai|Ri|Px|Bx)\b", "Rx", t)
        if n_rx:
            t = repaired_rx
            edit = edit or "rx_prefix_repair"
        if edit:
            return t, edit
        return t, None

    def _lexicon_repair_tokens(self, text: str) -> tuple[str, list[dict[str, str]]]:
        """Fuzzy-fix tokens that are already close to domain lexicon entries."""
        if not text or not self.domain_terms:
            return text, []
        parts = re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9]+", text)
        edits: list[dict[str, str]] = []
        out: list[str] = []
        for part in parts:
            if not re.fullmatch(r"[A-Za-z]+", part):
                out.append(part)
                continue
            low = part.lower()
            if low in self.domain_terms or low in ABBREV:
                canon = ABBREV.get(low, part)
                # Preserve all-caps / title from lexicon abbreviations
                if low in ABBREV:
                    out.append(ABBREV[low])
                    if ABBREV[low] != part:
                        edits.append({"from": part, "to": ABBREV[low], "type": "abbrev"})
                else:
                    out.append(part)
                continue
            match = process.extractOne(
                low,
                self.domain_terms,
                scorer=fuzz.ratio,
                score_cutoff=int((1.0 - self.max_token_dist) * 100),
            )
            if not match:
                out.append(part)
                continue
            term, score, _ = match
            dist = Levenshtein.normalized_distance(low, term)
            if dist > self.max_token_dist or score < 66:
                out.append(part)
                continue
            # Preserve capitalization style
            if part.isupper():
                replacement = term.upper()
            elif part[0].isupper():
                replacement = term.capitalize()
            else:
                replacement = term
            out.append(replacement)
            if replacement != part:
                edits.append(
                    {
                        "from": part,
                        "to": replacement,
                        "type": "lexicon_fuzzy",
                        "score": str(score),
                    }
                )
        return "".join(out), edits

    def _language_plausibility(self, text: str) -> float:
        if not text or text == "[ILLEGIBLE]":
            return 0.0
        score = 0.4
        if re.search(r"[A-Za-z]{2,}", text):
            score += 0.2
        if DATE_RE.match(text.replace(" ", "")) or QTY_RE.match(text):
            score += 0.3
        if text.lower() in self.domain_terms:
            score += 0.2
        # Penalize heavy punctuation / garbage
        if "?" in text or text.count(".") > 2:
            score -= 0.15
        if text.startswith("ink@"):
            score = 0.1
        return max(0.0, min(1.0, score))

    def _domain_score(self, text: str) -> float:
        low = text.lower()
        tokens = re.findall(r"[a-z0-9]+", low)
        if not tokens:
            return 0.0
        hits = sum(1 for t in tokens if t in self.domain_terms or t in self.units)
        if hits:
            return min(1.0, 0.35 + 0.2 * hits)
        if DATE_RE.match(text.replace(" ", "")):
            return 0.7
        return 0.0

    def _clone_hyp(
        self,
        h: Hypothesis,
        text: str,
        *,
        visual_boost: float = 0.0,
        note: Optional[dict[str, Any]] = None,
    ) -> Hypothesis:
        cfg = dict(h.configuration or {})
        if note:
            cfg["safe_edit"] = note
        return h.model_copy(
            update={
                "id": new_id(),
                "text": text,
                "normalized_text": normalize_candidate(text),
                "visual_score": float(min(0.95, h.visual_score + visual_boost)),
                "configuration": cfg,
            }
        )

    def rank(
        self,
        hypotheses: list[Hypothesis],
        context: Optional[dict[str, Any]] = None,
    ) -> RerankResult:
        context = context or {}
        if not hypotheses:
            return RerankResult(hypotheses=[], selected=None, context_score=0.0)

        # Expand with safe repairs derived from TrOCR hypotheses only
        expanded: list[Hypothesis] = list(hypotheses)
        for h in hypotheses:
            if not h.text or not is_trocr_hyp(h):
                continue
            formatted, edit_type = self._format_only(h.text)
            repaired, lex_edits = self._lexicon_repair_tokens(formatted)
            if repaired != h.text:
                note = {
                    "format_edit": edit_type,
                    "lexicon_edits": lex_edits,
                    "source_hypothesis_id": h.id,
                }
                # Small boost only when lexicon actually helped
                boost = 0.04 if lex_edits else 0.01
                expanded.append(self._clone_hyp(h, repaired, visual_boost=boost, note=note))

        scored: list[tuple[float, Hypothesis, Optional[dict[str, Any]]]] = []
        for h in expanded:
            formatted, edit_type = self._format_only(h.text)
            edit = None
            work = h
            if edit_type and formatted != h.text:
                work = self._clone_hyp(
                    h,
                    formatted,
                    note={
                        "from": h.text,
                        "to": formatted,
                        "edit_type": edit_type,
                        "source_hypothesis_id": h.id,
                    },
                )
                edit = {
                    "from": h.text,
                    "to": formatted,
                    "edit_type": edit_type,
                    "source_hypothesis_id": h.id,
                }

            min_dist = min(
                Levenshtein.normalized_distance(
                    work.normalized_text, o.normalized_text or ""
                )
                for o in hypotheses
            )
            # Reject free invention: repaired text must stay near some OCR candidate
            if min_dist > 0.45:
                continue

            lang = self._language_plausibility(work.text)
            domain = self._domain_score(work.text)
            disagreement = 0.0
            texts = {normalize_candidate(x.text) for x in hypotheses if x.text}
            if len(texts) > 3:
                disagreement = 0.15

            trocr_bonus = 0.08 if is_trocr_hyp(work) else -0.2
            score = (
                work.visual_score
                + 0.18 * lang
                + 0.16 * domain
                + trocr_bonus
                - 0.35 * min_dist
                - disagreement
            )
            scored.append((score, work, edit))

        if not scored:
            return RerankResult(
                hypotheses=hypotheses, selected=hypotheses[0], context_score=0.0
            )

        scored.sort(key=lambda x: -x[0])
        ranked = [s[1] for s in scored]
        best_edit = scored[0][2] or (ranked[0].configuration or {}).get("safe_edit")
        ctx_score = max(0.0, min(1.0, 0.35 * self._domain_score(ranked[0].text) + 0.2))
        return RerankResult(
            hypotheses=ranked,
            selected=ranked[0],
            format_edit=best_edit if isinstance(best_edit, dict) else None,
            context_score=ctx_score,
        )
