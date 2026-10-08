"""Document-processing orchestration — provenance-preserving vertical slice."""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from domain.entities import (
    AuditEvent,
    Decision,
    DecisionState,
    Document,
    JobStatus,
    Line,
    Page,
    QualityClass,
    RecognitionRun,
    Region,
    RegionType,
    ReviewTask,
    WordEvidence,
    new_id,
    utcnow,
)
from domain.config import load_thresholds
from domain.policy import DecisionEngine, DecisionThresholds, normalize_candidate
from infrastructure.imaging.preprocess import ImagePipeline
from infrastructure.models.context_reranker import ContextReranker
from infrastructure.models.detection import LineDetector
from infrastructure.models.recognition import (
    EnsembleRecognitionProvider,
    MockRecognitionProvider,
    RecognitionProvider,
    TrOCRRecognitionProvider,
)
from infrastructure.models.document_understanding import (
    DocumentUnderstanding,
    apply_layout_hints_to_lines,
    build_understanding_from_env,
    heuristic_understanding,
)
from infrastructure.models.verifier import VerifierProvider, build_verifier_from_env
from infrastructure.storage.artifact_store import ArtifactStore
from infrastructure.storage.repository import Repository

logger = logging.getLogger(__name__)

def _line_variants() -> tuple[str, ...]:
    # Fast default: CLAHE only. Hard lines escalate to multi-variant beams.
    raw = os.getenv(
        "SCRIBEPROOF_LINE_VARIANTS",
        "clahe",
    )
    variants = tuple(v.strip() for v in raw.split(",") if v.strip())
    return variants or ("clahe", "denoised")


def _escalate_variants() -> tuple[str, ...]:
    raw = os.getenv(
        "SCRIBEPROOF_ESCALATE_VARIANTS",
        "raw,clahe,adaptive_binarized,denoised",
    )
    return tuple(v.strip() for v in raw.split(",") if v.strip()) or (
        "raw",
        "clahe",
        "adaptive_binarized",
        "denoised",
    )


LINE_VARIANTS = _line_variants()


class DocumentProcessor:
    def __init__(
        self,
        repo: Repository,
        store: ArtifactStore,
        recognition: Optional[RecognitionProvider] = None,
        detector: Optional[LineDetector] = None,
        verifier: Optional[VerifierProvider] = None,
        thresholds: Optional[DecisionThresholds] = None,
    ):
        self.repo = repo
        self.store = store
        self.imaging = ImagePipeline()
        self.detector = detector or LineDetector(
            use_paddle=os.getenv("SCRIBEPROOF_USE_PADDLE", "true").lower()
            in {"1", "true", "yes"}
        )
        self.recognition = recognition or self._default_recognition()
        self.reranker = ContextReranker()
        self.verifier = verifier or build_verifier_from_env()
        self.understanding = build_understanding_from_env()
        self.engine = DecisionEngine(thresholds or load_thresholds())
        self.line_variants = _line_variants()

    def warmup_models(self) -> dict[str, Any]:
        """Load PaddleOCR + TrOCR so the first upload is not cold."""
        det_ok = False
        rec_ok = False
        try:
            det_ok = bool(self.detector.warmup())
        except Exception as e:
            logger.warning("Detector warmup failed: %s", e)
        try:
            rec_ok = bool(self.recognition.warmup())
        except Exception as e:
            logger.warning("Recognition warmup failed: %s", e)
        return {
            "detector": getattr(self.detector, "status", {"ready": det_ok}),
            "recognition": getattr(self.recognition, "status", {"ready": rec_ok}),
        }

    def model_status(self) -> dict[str, Any]:
        qwen_on = os.getenv("SCRIBEPROOF_QWEN_ENABLED", "false").lower() in {
            "1",
            "true",
            "yes",
        } and bool(os.getenv("SCRIBEPROOF_QWEN_API_KEY", "").strip())
        return {
            "detector": getattr(self.detector, "status", {}),
            "recognition": getattr(self.recognition, "status", {}),
            "line_variants": list(self.line_variants),
            "ocr_mode": os.getenv("SCRIBEPROOF_OCR_MODE", "trocr"),
            "use_paddle": os.getenv("SCRIBEPROOF_USE_PADDLE", "true"),
            "qwen_enabled": qwen_on,
            "qwen_model": os.getenv(
                "SCRIBEPROOF_QWEN_MODEL", "qwen/qwen3-vl-8b-instruct"
            ),
            "print_ocr_primary": os.getenv("SCRIBEPROOF_PRINT_OCR_PRIMARY", "true"),
        }

    def _default_recognition(self) -> RecognitionProvider:
        mode = os.getenv("SCRIBEPROOF_OCR_MODE", "trocr").lower()
        if mode == "mock":
            return MockRecognitionProvider()

        trocr = TrOCRRecognitionProvider()
        provider: RecognitionProvider = EnsembleRecognitionProvider(trocr)

        allow_mock = os.getenv("SCRIBEPROOF_ALLOW_MOCK_FALLBACK", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        if mode == "auto" and allow_mock:
            return _FallbackRecognition(provider, MockRecognitionProvider())
        return provider

    def create_document(
        self, filename: str, media_type: str, data: bytes
    ) -> Document:
        artifact = self.store.put_bytes_sync(
            data,
            media_type=media_type,
            transformation_name="raw_upload",
            settings={"filename": filename},
        )
        self.repo.save_artifact(artifact)
        doc = Document(
            filename=filename,
            media_type=media_type,
            source_artifact_id=artifact.id,
            status=JobStatus.PENDING,
        )
        # Link artifact to document
        artifact.document_id = doc.id
        self.store.write_meta(artifact)
        self.repo.save_artifact(artifact)
        self.repo.save_document(doc)
        job_id = new_id()
        self.repo.upsert_job(
            job_id, doc.id, "pending", "queued", 0.0, "Queued for processing"
        )
        return doc

    def process_document(self, document_id: str) -> None:
        doc = self.repo.get_document(document_id)
        if not doc:
            raise ValueError(f"Document {document_id} not found")
        job = self.repo.get_job_by_document(document_id)
        job_id = job["id"] if job else new_id()
        t0 = time.time()
        try:
            self.repo.update_document_status(document_id, JobStatus.PROCESSING)
            self._stage(job_id, document_id, "loading", 0.05, "Loading source artifact")

            source = self.repo.get_artifact(doc.source_artifact_id)
            if not source:
                raise ValueError("Source artifact missing")
            data = self.store.read_bytes(source)
            pages_bgr = self._load_pages(data, doc.media_type, doc.filename)

            self.repo.update_document_status(
                document_id, JobStatus.PROCESSING, page_count=len(pages_bgr)
            )

            all_words: list[WordEvidence] = []
            doc_understanding: Optional[DocumentUnderstanding] = None
            aggregated_line_items: list[dict[str, Any]] = []
            for page_index, page_img in enumerate(pages_bgr):
                progress_base = 0.1 + 0.8 * (page_index / max(len(pages_bgr), 1))
                self._stage(
                    job_id,
                    document_id,
                    "preprocessing",
                    progress_base,
                    f"Preprocessing page {page_index + 1}/{len(pages_bgr)}",
                )
                page, variants = self._process_page(
                    doc, source.id, page_index, page_img
                )

                self._stage(
                    job_id,
                    document_id,
                    "understanding",
                    progress_base + 0.08,
                    f"Layout + category page {page_index + 1}",
                )
                try:
                    page_understanding = self.understanding.understand(variants.deskewed)
                except Exception as e:
                    logger.warning("Document understanding failed: %s", e)
                    page_understanding = heuristic_understanding()
                    page_understanding.error = str(e)
                if doc_understanding is None:
                    doc_understanding = page_understanding
                else:
                    # Merge later pages' inventory into first understanding
                    doc_understanding.line_items.extend(page_understanding.line_items)
                    doc_understanding.layout_regions.extend(
                        page_understanding.layout_regions
                    )
                self.repo.save_audit(
                    AuditEvent(
                        document_id=document_id,
                        event_type="document_understanding",
                        payload={
                            "page_index": page_index,
                            "category": page_understanding.category.value,
                            "category_confidence": page_understanding.category_confidence,
                            "provider": page_understanding.provider,
                            "line_item_count": len(page_understanding.line_items),
                            "summary": page_understanding.summary,
                            "error": page_understanding.error,
                        },
                    )
                )

                self._stage(
                    job_id,
                    document_id,
                    "detection",
                    progress_base + 0.15,
                    f"Detecting lines on page {page_index + 1}",
                )
                det = self.detector.detect(variants.deskewed)
                page_h, page_w = variants.deskewed.shape[:2]
                # Strengthen offline heuristic once we know print vs handwriting ratios
                if not page_understanding.enabled and det.lines:
                    print_n = sum(
                        1
                        for L in det.lines
                        if (L.metadata or {}).get("script_mode") == "print"
                        or L.region_type == RegionType.PRINTED_TEXT
                    )
                    ratio = print_n / max(len(det.lines), 1)
                    page_understanding = heuristic_understanding(
                        paddle_print_ratio=ratio, line_count=len(det.lines)
                    )
                    if doc_understanding is None or not doc_understanding.enabled:
                        doc_understanding = page_understanding
                apply_layout_hints_to_lines(
                    det.lines, page_understanding, page_w, page_h
                )

                region_cache: dict[str, Region] = {}
                lines_entities: list[Line] = []

                for order, dline in enumerate(det.lines):
                    rt = dline.region_type
                    key = f"{rt.value}:{order}"  # one region shell per line for accurate type
                    if key not in region_cache:
                        region = Region(
                            page_id=page.id,
                            region_type=rt,
                            bbox=dline.bbox,
                            metadata={
                                "detector": det.detector_name,
                                "script_mode": dline.metadata.get("script_mode"),
                                "layout_source": dline.metadata.get("layout_source"),
                            },
                        )
                        # Crop region
                        crop = self.imaging.crop_bbox(
                            variants.deskewed,
                            dline.bbox.x,
                            dline.bbox.y,
                            dline.bbox.width,
                            dline.bbox.height,
                        )
                        crop_art = self.store.put_bytes_sync(
                            self.imaging.encode_png(crop),
                            media_type="image/png",
                            transformation_name="region_crop",
                            settings={"region_type": rt.value},
                            source_artifact_id=page.artifact_ids.get("deskewed"),
                            page_id=page.id,
                            document_id=document_id,
                        )
                        self.repo.save_artifact(crop_art)
                        region.crop_artifact_id = crop_art.id
                        self.repo.save_region(region)
                        region_cache[key] = region
                    else:
                        region = region_cache[key]

                    # Link margin notes to nearest body region
                    if rt == RegionType.MARGIN_NOTE:
                        body = next(
                            (
                                r
                                for r in region_cache.values()
                                if r.region_type
                                in {
                                    RegionType.MAIN_HANDWRITING,
                                    RegionType.PRINTED_TEXT,
                                    RegionType.TABLE,
                                }
                            ),
                            None,
                        )
                        if body:
                            region.linked_region_id = body.id
                            self.repo.save_region(region)

                    crop_ids: dict[str, str] = {}
                    variant_images = {
                        "raw": variants.deskewed,
                        "clahe": variants.clahe,
                        "adaptive_binarized": variants.adaptive_binarized,
                        "denoised": variants.denoised,
                    }
                    for vname, vimg in variant_images.items():
                        crop = self.imaging.crop_bbox(
                            vimg,
                            dline.bbox.x,
                            dline.bbox.y,
                            dline.bbox.width,
                            dline.bbox.height,
                        )
                        art = self.store.put_bytes_sync(
                            self.imaging.encode_png(crop),
                            media_type="image/png",
                            transformation_name=f"line_crop_{vname}",
                            settings={
                                "variant": vname,
                                "bbox": dline.bbox.as_dict(),
                                "reading_order": order,
                            },
                            source_artifact_id=page.artifact_ids.get(vname)
                            or page.artifact_ids.get("deskewed"),
                            page_id=page.id,
                            document_id=document_id,
                        )
                        self.repo.save_artifact(art)
                        crop_ids[vname] = art.id

                    line = Line(
                        page_id=page.id,
                        region_id=region.id,
                        bbox=dline.bbox,
                        reading_order=order,
                        polygon=dline.polygon,
                        crop_artifact_ids=crop_ids,
                        is_crossed_out_candidate=dline.is_crossed_out_candidate,
                        crossed_out_score=dline.crossed_out_score,
                        metadata={
                            "detector": det.detector_name,
                            "detection_confidence": dline.confidence,
                            **dline.metadata,
                        },
                    )
                    self.repo.save_line(line)
                    lines_entities.append(line)

                self._stage(
                    job_id,
                    document_id,
                    "recognition",
                    progress_base + 0.35,
                    f"Recognizing {len(lines_entities)} lines",
                )

                quality_risk = {
                    "good": 0.1,
                    "fair": 0.3,
                    "poor": 0.55,
                    "unusable": 0.85,
                }.get(
                    page.quality_class.value if page.quality_class else "fair", 0.3
                )

                for line_idx, line in enumerate(lines_entities):
                    all_hyps = []
                    evidence_ids = []
                    paddle_text = line.metadata.get("paddle_text") or ""
                    paddle_score = line.metadata.get("paddle_score")
                    script_mode = (line.metadata.get("script_mode") or "handwriting").lower()
                    # Resolve region type for this line
                    line_region_type = RegionType.MAIN_HANDWRITING
                    for r in self.repo.list_regions(page.id):
                        if r.id == line.region_id:
                            line_region_type = r.region_type
                            break
                    if line_region_type == RegionType.PRINTED_TEXT:
                        script_mode = "print"
                    elif line_region_type == RegionType.TABLE:
                        script_mode = "print"
                    print_mode = script_mode == "print"
                    self._stage(
                        job_id,
                        document_id,
                        "recognition",
                        progress_base
                        + 0.35
                        + 0.4 * (line_idx / max(len(lines_entities), 1)),
                        (
                            f"Print OCR line {line_idx + 1}/{len(lines_entities)}"
                            if print_mode
                            else f"TrOCR line {line_idx + 1}/{len(lines_entities)}"
                        ),
                    )

                    def _run_variants(variant_names: tuple[str, ...], *, fast: bool) -> None:
                        nonlocal all_hyps, evidence_ids
                        for vname in variant_names:
                            crop_id = line.crop_artifact_ids.get(vname)
                            if not crop_id:
                                continue
                            # Skip already-recognized variants on escalate
                            if any(h.image_variant == vname for h in all_hyps):
                                continue
                            crop_art = self.repo.get_artifact(crop_id)
                            if not crop_art:
                                continue
                            crop_img = self.imaging.load_from_bytes(
                                self.store.read_bytes(crop_art)
                            )
                            run = RecognitionRun(
                                line_id=line.id,
                                provider=type(self.recognition).__name__,
                                model_name=getattr(
                                    self.recognition, "MODEL_NAME", "unknown"
                                ),
                                model_version=getattr(
                                    self.recognition, "MODEL_VERSION", "unknown"
                                ),
                                image_variant=vname,
                                source_crop_id=crop_id,
                                configuration={
                                    "paddle_text": paddle_text,
                                    "paddle_score": paddle_score,
                                    "fast": fast,
                                    "script_mode": script_mode,
                                    "region_type": line_region_type.value,
                                },
                            )
                            self.repo.save_recognition_run(run)
                            hyps = self.recognition.recognize(
                                crop_img,
                                {
                                    "recognition_run_id": run.id,
                                    "source_crop_id": crop_id,
                                    "image_variant": vname,
                                    "paddle_text": paddle_text,
                                    "paddle_score": paddle_score,
                                    "fast": fast,
                                    "script_mode": script_mode,
                                    "region_type": line_region_type.value,
                                },
                            )
                            for h in hyps:
                                h.recognition_run_id = run.id
                                h.source_crop_id = crop_id
                                h.image_variant = vname
                                self.repo.save_hypothesis(h, line.id)
                                all_hyps.append(h)
                                evidence_ids.append(h.id)

                    # Pass 1 — fast path (print usually one variant; handwriting CLAHE)
                    _run_variants(tuple(self.line_variants), fast=True)

                    # Pass 2 — escalate hard handwriting lines only
                    trocr_hyps = [
                        h
                        for h in all_hyps
                        if h.text
                        and (
                            "trocr" in (h.model_name or "").lower()
                            or "microsoft" in (h.model_name or "").lower()
                        )
                    ]
                    paddle_hyps = [
                        h
                        for h in all_hyps
                        if h.text and "paddle" in (h.model_name or "").lower()
                    ]
                    trocr_scores = [h.visual_score for h in trocr_hyps]
                    max_trocr = max(trocr_scores) if trocr_scores else 0.0
                    max_paddle = (
                        max(h.visual_score for h in paddle_hyps) if paddle_hyps else 0.0
                    )
                    _domain = {
                        "patient",
                        "allergy",
                        "nkda",
                        "amoxicillin",
                        "rx",
                        "mg",
                        "notes",
                        "date",
                        "follow",
                        "fever",
                        "tid",
                        "bid",
                        "dose",
                        "days",
                        "take",
                        "food",
                        "jordan",
                        "miles",
                    }
                    best_coverage = 0.0
                    for h in trocr_hyps:
                        toks = [
                            t.lower()
                            for t in __import__("re").findall(r"[A-Za-z]+", h.text)
                            if len(t) > 1
                        ]
                        if not toks:
                            continue
                        hits = sum(1 for t in toks if t in _domain)
                        best_coverage = max(best_coverage, hits / len(toks))
                    needs_escalate = False
                    if not print_mode:
                        needs_escalate = max_trocr < 0.50 or best_coverage <= 0.25
                    elif max_paddle < 0.55 and max_trocr < 0.45:
                        needs_escalate = True
                    if needs_escalate:
                        self._stage(
                            job_id,
                            document_id,
                            "recognition",
                            progress_base
                            + 0.35
                            + 0.4 * (line_idx / max(len(lines_entities), 1)),
                            f"Escalate hard line {line_idx + 1}/{len(lines_entities)}",
                        )
                        _run_variants(_escalate_variants(), fast=False)

                    # Phase 5: safe contextual rerank
                    rerank = self.reranker.rank(
                        all_hyps,
                        context={"page_quality": page.quality_class},
                    )
                    if rerank.format_edit:
                        self.repo.save_audit(
                            AuditEvent(
                                document_id=document_id,
                                event_type="format_edit",
                                payload=rerank.format_edit,
                            )
                        )
                    ranked_hyps = rerank.hypotheses or all_hyps

                    # Phase 6: optional verifier for low-confidence / high-risk
                    verifier_agreement = 0.0
                    verifier_disagreement = 0.0
                    prelim_visual = (
                        max((h.visual_score for h in ranked_hyps), default=0.0)
                    )
                    if prelim_visual < 0.55 and ranked_hyps:
                        crop_id = line.crop_artifact_ids.get("clahe") or line.crop_artifact_ids.get(
                            "raw"
                        )
                        if crop_id:
                            crop_art = self.repo.get_artifact(crop_id)
                            if crop_art:
                                crop_img = self.imaging.load_from_bytes(
                                    self.store.read_bytes(crop_art)
                                )
                                vres = self.verifier.verify(
                                    crop_img, [h.text for h in ranked_hyps[:5]]
                                )
                                self.repo.save_audit(
                                    AuditEvent(
                                        document_id=document_id,
                                        event_type="verifier_result",
                                        payload={
                                            "line_id": line.id,
                                            "status": vres.status,
                                            "reason": vres.reason,
                                            "selected_candidate_index": vres.selected_candidate_index,
                                            "enabled": vres.enabled,
                                        },
                                    )
                                )
                                if vres.enabled and vres.status == "supported":
                                    verifier_agreement = 1.0
                                elif vres.enabled and vres.status in {
                                    "illegible",
                                    "disagreement",
                                }:
                                    verifier_disagreement = 1.0

                    fusion = self.engine.fuse(
                        ranked_hyps,
                        is_crossed_out=line.is_crossed_out_candidate,
                        image_quality_risk=quality_risk,
                        context_score=rerank.context_score,
                        verifier_agreement=verifier_agreement,
                        verifier_disagreement=verifier_disagreement,
                        script_mode=script_mode,
                        region_type=line_region_type.value,
                    )

                    region_type = line_region_type

                    we = WordEvidence(
                        line_id=line.id,
                        page_id=page.id,
                        document_id=document_id,
                        selected_hypothesis_id=fusion.selected.id
                        if fusion.selected
                        else None,
                        text=fusion.text,
                        decision_state=fusion.state,
                        confidence=fusion.confidence,
                        uncertainty=fusion.uncertainty,
                        final_score=fusion.final_score,
                        alternatives=fusion.alternatives,
                        evidence_ids=evidence_ids,
                        crop_artifact_id=line.crop_artifact_ids.get("clahe")
                        or line.crop_artifact_ids.get("raw"),
                        decision_reason=fusion.reason,
                        is_high_risk_entity=fusion.is_high_risk,
                        reading_order=line.reading_order,
                        bbox=line.bbox,
                        region_type=region_type,
                        score_breakdown={
                            "normalized_visual_score": fusion.breakdown.normalized_visual_score,
                            "variant_agreement": fusion.breakdown.variant_agreement,
                            "character_stability": fusion.breakdown.character_stability,
                            "context_score": fusion.breakdown.context_score,
                            "verifier_agreement": fusion.breakdown.verifier_agreement,
                            "decoder_entropy": fusion.breakdown.decoder_entropy,
                            "ensemble_disagreement": fusion.breakdown.ensemble_disagreement,
                            "image_quality_risk": fusion.breakdown.image_quality_risk,
                            "verifier_disagreement": fusion.breakdown.verifier_disagreement,
                            "final_score": fusion.breakdown.final_score,
                            "uncertainty": fusion.breakdown.uncertainty,
                        },
                    )
                    self.repo.save_word_evidence(we)
                    decision = Decision(
                        word_evidence_id=we.id,
                        state=fusion.state,
                        confidence=fusion.confidence,
                        uncertainty=fusion.uncertainty,
                        reason=fusion.reason,
                        thresholds_used=self.engine.thresholds.as_numeric_dict(),
                        score_breakdown=we.score_breakdown,
                    )
                    self.repo.save_decision(decision)
                    if fusion.state in {
                        DecisionState.REVIEW_REQUIRED,
                        DecisionState.ILLEGIBLE,
                    }:
                        self.repo.save_review_task(
                            ReviewTask(
                                document_id=document_id,
                                word_evidence_id=we.id,
                                priority=2 if fusion.is_high_risk else 1,
                            )
                        )
                    all_words.append(we)
                    if (
                        fusion.text
                        and fusion.state != DecisionState.CROSSED_OUT
                        and fusion.text != "[ILLEGIBLE]"
                    ):
                        aggregated_line_items.append(
                            {
                                "id": we.id,
                                "label": f"line_{line.reading_order + 1}",
                                "value": fusion.text,
                                "kind": "ocr_line",
                                "source": "ocr",
                                "confidence": float(fusion.confidence),
                                "decision_state": fusion.state.value,
                                "region_type": region_type.value,
                                "script_mode": script_mode,
                            }
                        )

            # Persist category / layout / inventory on document metadata
            if doc_understanding is None:
                doc_understanding = heuristic_understanding(
                    line_count=len(all_words)
                )
            meta = doc_understanding.as_metadata()
            # Prefer Qwen inventory first, then OCR lines as separated items
            qwen_items = meta.get("line_items") or []
            merged_items = list(qwen_items) + aggregated_line_items
            meta["line_items"] = merged_items
            meta["ocr_line_count"] = len(aggregated_line_items)
            doc.metadata = {**(doc.metadata or {}), **meta}
            doc.status = JobStatus.COMPLETED
            doc.page_count = len(pages_bgr)
            doc.updated_at = utcnow()
            self.repo.save_document(doc)

            elapsed = time.time() - t0
            self.repo.save_audit(
                AuditEvent(
                    document_id=document_id,
                    event_type="processing_complete",
                    payload={
                        "elapsed_seconds": elapsed,
                        "word_count": len(all_words),
                        "accepted": sum(
                            1
                            for w in all_words
                            if w.decision_state == DecisionState.ACCEPTED
                        ),
                        "category": meta.get("category"),
                        "line_item_count": len(merged_items),
                    },
                )
            )
            self.repo.update_document_status(document_id, JobStatus.COMPLETED)
            self._stage(
                job_id,
                document_id,
                "completed",
                1.0,
                f"Completed in {elapsed:.1f}s",
            )
        except Exception as e:
            logger.exception("Processing failed for %s", document_id)
            self.repo.update_document_status(
                document_id, JobStatus.FAILED, error_message=str(e)
            )
            self._stage(
                job_id, document_id, "failed", 1.0, "Failed", error_message=str(e)
            )
            raise

    def _stage(
        self,
        job_id: str,
        document_id: str,
        stage: str,
        progress: float,
        message: str,
        error_message: Optional[str] = None,
    ) -> None:
        status = "failed" if stage == "failed" else (
            "completed" if stage == "completed" else "processing"
        )
        self.repo.upsert_job(
            job_id,
            document_id,
            status,
            stage,
            progress,
            message,
            error_message=error_message,
        )

    def _load_pages(
        self, data: bytes, media_type: str, filename: str
    ) -> list[np.ndarray]:
        lower = filename.lower()
        if media_type == "application/pdf" or lower.endswith(".pdf"):
            return self._pdf_to_images(data)
        img = self.imaging.load_with_exif(data)
        return [img]

    def _pdf_to_images(self, data: bytes) -> list[np.ndarray]:
        try:
            from pdf2image import convert_from_bytes

            pil_pages = convert_from_bytes(data, dpi=200)
            out = []
            for pil in pil_pages:
                rgb = np.array(pil.convert("RGB"))
                out.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if out:
                return out
        except Exception as e:
            logger.warning("pdf2image failed (%s); trying PyMuPDF-less PIL fallback", e)
        # Minimal fallback: if PDF tooling missing, raise clearly
        raise RuntimeError(
            "PDF conversion requires poppler (pdf2image). "
            "Install poppler or upload PNG/JPG instead."
        )

    def _process_page(
        self, doc: Document, source_artifact_id: str, page_index: int, page_img: np.ndarray
    ) -> tuple[Page, Any]:
        bundle = self.imaging.process_page(page_img)
        h, w = bundle.raw.shape[:2]

        def store_variant(name: str, img: np.ndarray, source_id: Optional[str]) -> str:
            art = self.store.put_bytes_sync(
                self.imaging.encode_png(img),
                media_type="image/png",
                transformation_name=name,
                settings=bundle.settings,
                source_artifact_id=source_id,
                document_id=doc.id,
            )
            self.repo.save_artifact(art)
            return art.id

        raw_id = store_variant("raw", bundle.raw, source_artifact_id)
        normalized_id = store_variant("normalized", bundle.normalized, raw_id)
        deskewed_id = store_variant("deskewed", bundle.deskewed, normalized_id)
        clahe_id = store_variant("clahe", bundle.clahe, deskewed_id)
        adaptive_id = store_variant(
            "adaptive_binarized", bundle.adaptive_binarized, deskewed_id
        )
        denoised_id = store_variant("denoised", bundle.denoised, deskewed_id)

        page = Page(
            document_id=doc.id,
            page_index=page_index,
            width=w,
            height=h,
            raw_artifact_id=raw_id,
            normalized_artifact_id=normalized_id,
            blur_score=bundle.quality.blur_score,
            contrast_score=bundle.quality.contrast_score,
            skew_angle=bundle.quality.skew_angle,
            resolution=bundle.quality.resolution,
            noise_score=bundle.quality.noise_score,
            quality_class=QualityClass(bundle.quality.quality_class),
            artifact_ids={
                "raw": raw_id,
                "normalized": normalized_id,
                "deskewed": deskewed_id,
                "clahe": clahe_id,
                "adaptive_binarized": adaptive_id,
                "denoised": denoised_id,
            },
            preprocessing_settings=bundle.settings,
        )
        # Fix page_id on artifacts (already stored; update meta for inspectability)
        for aid in page.artifact_ids.values():
            a = self.repo.get_artifact(aid)
            if a:
                a.page_id = page.id
                self.repo.save_artifact(a)
        self.repo.save_page(page)
        return page, bundle


class _FallbackRecognition(RecognitionProvider):
    def __init__(self, primary: RecognitionProvider, fallback: RecognitionProvider):
        self.primary = primary
        self.fallback = fallback
        self._use_fallback = False
        self.MODEL_NAME = getattr(primary, "MODEL_NAME", "auto")
        self.MODEL_VERSION = getattr(primary, "MODEL_VERSION", "auto")

    def recognize(self, image, options=None):
        if self._use_fallback:
            return self.fallback.recognize(image, options)
        try:
            hyps = self.primary.recognize(image, options)
            # If primary returned empty-score failure marker
            if (
                hyps
                and not hyps[0].text
                and hyps[0].configuration.get("error")
            ):
                self._use_fallback = True
                self.MODEL_NAME = getattr(self.fallback, "MODEL_NAME", "mock")
                return self.fallback.recognize(image, options)
            return hyps
        except Exception as e:
            logger.warning("Primary OCR failed (%s); using fallback", e)
            self._use_fallback = True
            self.MODEL_NAME = getattr(self.fallback, "MODEL_NAME", "mock")
            return self.fallback.recognize(image, options)
