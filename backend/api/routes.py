from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from api.schemas import (
    BoundingBoxSchema,
    DocumentCreateResponse,
    DocumentGraphSchema,
    DocumentMetaResponse,
    DocumentResultResponse,
    DocumentStatusResponse,
    HealthResponse,
    HypothesisSchema,
    LayoutRegionSchema,
    LineItemSchema,
    LineSchema,
    PageSchema,
    RegionSchema,
    WordPatchRequest,
    WordPatchResponse,
    WordResultSchema,
)
from domain.entities import CorrectionRecord, DecisionState, JobStatus, utcnow
from domain.policy import normalize_candidate

router = APIRouter()


def get_deps():
    from main import get_processor, get_repo, get_store

    return get_repo(), get_store(), get_processor()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", service="scribeproof-api", version="1.0.0")


@router.get("/api/models/status")
def models_status():
    """Report PaddleOCR detection + TrOCR recognition readiness."""
    try:
        _, _, processor = get_deps()
        return {"status": "ok", **processor.model_status()}
    except Exception as e:
        return {"status": "starting", "error": str(e)}


@router.post("/api/documents", response_model=DocumentCreateResponse)
async def create_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    repo, store, processor = get_deps()
    filename = file.filename or "upload.bin"
    media_type = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    allowed = {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "application/pdf",
    }
    ext = Path(filename).suffix.lower()
    if media_type not in allowed and ext not in {".png", ".jpg", ".jpeg", ".pdf"}:
        raise HTTPException(400, "Only PNG, JPG, JPEG, or PDF uploads are supported")
    if ext == ".jpg":
        media_type = "image/jpeg"
    if ext == ".png":
        media_type = "image/png"
    if ext == ".pdf":
        media_type = "application/pdf"

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")

    doc = processor.create_document(filename, media_type, data)
    source = repo.get_artifact(doc.source_artifact_id)
    background_tasks.add_task(processor.process_document, doc.id)
    return DocumentCreateResponse(
        document_id=doc.id,
        status=doc.status.value,
        filename=doc.filename,
        source_artifact_id=doc.source_artifact_id,
        sha256=source.sha256 if source else "",
    )


@router.get("/api/documents/{document_id}", response_model=DocumentMetaResponse)
def get_document(document_id: str):
    repo, _, _ = get_deps()
    doc = repo.get_document(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    meta = doc.metadata or {}
    return DocumentMetaResponse(
        id=doc.id,
        filename=doc.filename,
        media_type=doc.media_type,
        status=doc.status.value,
        page_count=doc.page_count,
        source_artifact_id=doc.source_artifact_id,
        created_at=doc.created_at.isoformat(),
        updated_at=doc.updated_at.isoformat(),
        error_message=doc.error_message,
        category=meta.get("category"),
        category_confidence=meta.get("category_confidence"),
    )


@router.get("/api/documents/{document_id}/status", response_model=DocumentStatusResponse)
def get_status(document_id: str):
    repo, _, _ = get_deps()
    doc = repo.get_document(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    job = repo.get_job_by_document(document_id)
    return DocumentStatusResponse(
        document_id=document_id,
        status=doc.status.value,
        stage=job["stage"] if job else "unknown",
        progress=job["progress"] if job else 0.0,
        message=job["message"] if job else "",
        error_message=doc.error_message or (job.get("error_message") if job else None),
        page_count=doc.page_count,
    )


@router.get("/api/documents/{document_id}/result", response_model=DocumentResultResponse)
def get_result(document_id: str):
    repo, _, _ = get_deps()
    doc = repo.get_document(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    if doc.status != JobStatus.COMPLETED:
        raise HTTPException(
            409,
            f"Document not ready (status={doc.status.value}). Poll /status.",
        )

    pages = repo.list_pages(document_id)
    words = repo.list_word_evidence(document_id)
    page_schemas: list[PageSchema] = []
    graph_nodes = []
    graph_edges = []
    hyps_by_line: dict[str, list[HypothesisSchema]] = {}

    for page in pages:
        regions = repo.list_regions(page.id)
        lines = repo.list_lines(page.id)
        region_type_map = {r.id: r.region_type.value for r in regions}
        line_schemas = []
        prev_node = None
        for line in lines:
            ls = LineSchema(
                id=line.id,
                reading_order=line.reading_order,
                bbox=BoundingBoxSchema(**line.bbox.as_dict()),
                polygon=line.polygon,
                region_id=line.region_id,
                is_crossed_out_candidate=line.is_crossed_out_candidate,
                crossed_out_score=line.crossed_out_score,
                crop_artifact_ids=line.crop_artifact_ids,
                region_type=region_type_map.get(line.region_id),
            )
            line_schemas.append(ls)
            node = {
                "id": line.id,
                "type": "line",
                "reading_order": line.reading_order,
                "region_type": region_type_map.get(line.region_id),
                "bbox": line.bbox.as_dict(),
            }
            graph_nodes.append(node)
            if prev_node is not None:
                graph_edges.append(
                    {"from": prev_node, "to": line.id, "relation": "reading_order"}
                )
            # Margin links
            if region_type_map.get(line.region_id) == "margin_note":
                for r in regions:
                    if r.id == line.region_id and r.linked_region_id:
                        graph_edges.append(
                            {
                                "from": line.id,
                                "to": r.linked_region_id,
                                "relation": "margin_to_body",
                            }
                        )
            prev_node = line.id

            hyps = repo.list_hypotheses_for_line(line.id)
            hyps_by_line[line.id] = [
                HypothesisSchema(
                    id=h.id,
                    text=h.text,
                    normalized_text=h.normalized_text,
                    rank=h.rank,
                    sequence_score=h.sequence_score,
                    visual_score=h.visual_score,
                    model_name=h.model_name,
                    model_version=h.model_version,
                    source_crop_id=h.source_crop_id,
                    image_variant=h.image_variant,
                )
                for h in hyps
            ]

        page_schemas.append(
            PageSchema(
                id=page.id,
                page_index=page.page_index,
                width=page.width,
                height=page.height,
                quality={
                    "blur_score": page.blur_score,
                    "contrast_score": page.contrast_score,
                    "skew_angle": page.skew_angle,
                    "resolution": page.resolution,
                    "noise_score": page.noise_score,
                    "quality_class": page.quality_class.value
                    if page.quality_class
                    else None,
                },
                artifact_ids=page.artifact_ids,
                preprocessing_settings=page.preprocessing_settings,
                regions=[
                    RegionSchema(
                        id=r.id,
                        region_type=r.region_type.value,
                        bbox=BoundingBoxSchema(**r.bbox.as_dict()),
                        linked_region_id=r.linked_region_id,
                        crop_artifact_id=r.crop_artifact_id,
                    )
                    for r in regions
                ],
                lines=line_schemas,
            )
        )

    def to_word(w) -> WordResultSchema:
        return WordResultSchema(
            id=w.id,
            text=w.text,
            decision_state=w.decision_state.value,
            confidence=w.confidence,
            uncertainty=w.uncertainty,
            final_score=w.final_score,
            alternatives=w.alternatives,
            evidence_ids=w.evidence_ids,
            crop_artifact_id=w.crop_artifact_id,
            decision_reason=w.decision_reason,
            is_high_risk_entity=w.is_high_risk_entity,
            reading_order=w.reading_order,
            bbox=BoundingBoxSchema(**w.bbox.as_dict()) if w.bbox else None,
            region_type=w.region_type.value,
            score_breakdown=w.score_breakdown,
            line_id=w.line_id,
            page_id=w.page_id,
            selected_hypothesis_id=w.selected_hypothesis_id,
        )

    active = [
        w
        for w in words
        if w.decision_state != DecisionState.CROSSED_OUT
    ]
    crossed = [w for w in words if w.decision_state == DecisionState.CROSSED_OUT]
    active_text = "\n".join(w.text for w in active)
    md_lines = ["# ScribeProof Transcription", ""]
    for w in words:
        state = w.decision_state.value
        if state == "CROSSED_OUT":
            md_lines.append(f"~~{w.text}~~ <!-- crossed-out uncertainty={w.uncertainty:.2f} -->")
        elif state == "ILLEGIBLE":
            md_lines.append(f"**[ILLEGIBLE]** <!-- uncertainty={w.uncertainty:.2f} -->")
        elif state == "REVIEW_REQUIRED":
            md_lines.append(f"=={w.text}== <!-- review conf={w.confidence:.2f} -->")
        else:
            md_lines.append(w.text)
    markdown = "\n".join(md_lines)

    meta = doc.metadata or {}
    layout_summary = [
        LayoutRegionSchema(
            region_type=str(r.get("region_type") or "unknown"),
            description=str(r.get("description") or ""),
            bbox_norm=r.get("bbox_norm"),
            raw_type=r.get("raw_type"),
        )
        for r in (meta.get("layout_summary") or [])
        if isinstance(r, dict)
    ]
    line_items = [
        LineItemSchema(
            id=str(it.get("id") or ""),
            label=str(it.get("label") or "item"),
            value=str(it.get("value") or ""),
            kind=str(it.get("kind") or "field"),
            source=str(it.get("source") or "ocr"),
            confidence=float(it.get("confidence") or 0.0),
            decision_state=it.get("decision_state"),
            region_type=it.get("region_type"),
            script_mode=it.get("script_mode"),
        )
        for it in (meta.get("line_items") or [])
        if isinstance(it, dict)
    ]
    # Fallback inventory from OCR words if metadata empty
    if not line_items:
        line_items = [
            LineItemSchema(
                id=w.id,
                label=f"line_{w.reading_order + 1}",
                value=w.text,
                kind="ocr_line",
                source="ocr",
                confidence=w.confidence,
                decision_state=w.decision_state.value,
                region_type=w.region_type.value,
            )
            for w in active
        ]

    return DocumentResultResponse(
        document_id=doc.id,
        filename=doc.filename,
        status=doc.status.value,
        pages=page_schemas,
        words=[to_word(w) for w in words],
        crossed_out_words=[to_word(w) for w in crossed],
        active_text=active_text,
        markdown=markdown,
        graph=DocumentGraphSchema(nodes=graph_nodes, edges=graph_edges),
        hypotheses_by_line=hyps_by_line,
        category=str(meta.get("category") or "unknown"),
        category_confidence=float(meta.get("category_confidence") or 0.0),
        summary=str(meta.get("summary") or ""),
        layout_summary=layout_summary,
        line_items=line_items,
    )


@router.patch("/api/words/{word_id}", response_model=WordPatchResponse)
def patch_word(word_id: str, body: WordPatchRequest):
    repo, _, _ = get_deps()
    word = repo.get_word_evidence(word_id)
    if not word:
        raise HTTPException(404, "Word not found")

    prev_text = word.text
    prev_state = word.decision_state
    new_text = body.text if body.text is not None else word.text
    if body.decision_state:
        try:
            new_state = DecisionState(body.decision_state)
        except ValueError:
            raise HTTPException(400, "Invalid decision_state")
    else:
        new_state = DecisionState.ACCEPTED if body.text is not None else word.decision_state

    word.text = normalize_candidate(new_text) if new_text != "[ILLEGIBLE]" else new_text
    word.decision_state = new_state
    word.updated_at = utcnow()
    repo.save_word_evidence(word)

    correction = CorrectionRecord(
        word_evidence_id=word.id,
        previous_text=prev_text,
        new_text=word.text,
        previous_state=prev_state,
        new_state=new_state,
        actor=body.actor,
    )
    repo.save_correction(correction)

    return WordPatchResponse(
        word_id=word.id,
        text=word.text,
        decision_state=word.decision_state.value,
        correction_id=correction.id,
    )


@router.get("/api/artifacts/{artifact_id}")
def get_artifact(artifact_id: str):
    repo, store, _ = get_deps()
    artifact = repo.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(404, "Artifact not found")
    path = Path(artifact.path)
    if not path.exists():
        raise HTTPException(404, "Artifact blob missing")
    return FileResponse(
        path,
        media_type=artifact.media_type,
        headers={
            "X-Artifact-SHA256": artifact.sha256,
            "X-Transformation": artifact.transformation_name,
            "X-Source-Artifact-Id": artifact.source_artifact_id or "",
        },
    )


@router.get("/api/artifacts/{artifact_id}/meta")
def get_artifact_meta(artifact_id: str):
    repo, _, _ = get_deps()
    artifact = repo.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(404, "Artifact not found")
    return artifact.model_dump(mode="json")


@router.get("/api/documents/{document_id}/export/{fmt}")
def export_document(document_id: str, fmt: str):
    repo, _, _ = get_deps()
    doc = repo.get_document(document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    if doc.status != JobStatus.COMPLETED:
        raise HTTPException(409, "Document not ready")

    # Reuse result builder
    result = get_result(document_id)
    if fmt == "txt":
        return Response(result.active_text, media_type="text/plain")
    if fmt == "md":
        return Response(result.markdown, media_type="text/markdown")
    if fmt == "json":
        return Response(
            result.model_dump_json(indent=2),
            media_type="application/json",
        )
    raise HTTPException(400, "fmt must be txt, md, or json")
