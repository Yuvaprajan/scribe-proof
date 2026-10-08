"""SQLite repository with PostgreSQL-compatible schema design."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from domain.entities import (
    Artifact,
    AuditEvent,
    CorrectionRecord,
    Decision,
    DecisionState,
    Document,
    Hypothesis,
    JobStatus,
    Line,
    Page,
    QualityClass,
    RecognitionRun,
    Region,
    RegionType,
    ReviewTask,
    WordEvidence,
    BoundingBox,
    utcnow,
)

Base = declarative_base()
metadata = MetaData()


class DocumentRow(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    media_type = Column(String, nullable=False)
    source_artifact_id = Column(String, nullable=False)
    status = Column(String, nullable=False)
    page_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    metadata_json = Column(Text, default="{}")


class JobRow(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True)
    document_id = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False)
    stage = Column(String, default="queued")
    progress = Column(Float, default=0.0)
    message = Column(Text, default="")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class ArtifactRow(Base):
    __tablename__ = "artifacts"
    id = Column(String, primary_key=True)
    sha256 = Column(String, nullable=False, index=True)
    path = Column(String, nullable=False)
    media_type = Column(String, nullable=False)
    source_artifact_id = Column(String, nullable=True)
    transformation_name = Column(String, nullable=False)
    settings_json = Column(Text, default="{}")
    created_at = Column(DateTime, nullable=False)
    page_id = Column(String, nullable=True)
    document_id = Column(String, nullable=True)


class PageRow(Base):
    __tablename__ = "pages"
    id = Column(String, primary_key=True)
    document_id = Column(String, nullable=False, index=True)
    page_index = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    raw_artifact_id = Column(String, nullable=False)
    normalized_artifact_id = Column(String, nullable=True)
    blur_score = Column(Float, nullable=True)
    contrast_score = Column(Float, nullable=True)
    skew_angle = Column(Float, nullable=True)
    resolution = Column(Float, nullable=True)
    noise_score = Column(Float, nullable=True)
    quality_class = Column(String, nullable=True)
    artifact_ids_json = Column(Text, default="{}")
    preprocessing_settings_json = Column(Text, default="{}")
    created_at = Column(DateTime, nullable=False)


class RegionRow(Base):
    __tablename__ = "regions"
    id = Column(String, primary_key=True)
    page_id = Column(String, nullable=False, index=True)
    region_type = Column(String, nullable=False)
    bbox_json = Column(Text, nullable=False)
    linked_region_id = Column(String, nullable=True)
    crop_artifact_id = Column(String, nullable=True)
    metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime, nullable=False)


class LineRow(Base):
    __tablename__ = "lines"
    id = Column(String, primary_key=True)
    page_id = Column(String, nullable=False, index=True)
    region_id = Column(String, nullable=False)
    bbox_json = Column(Text, nullable=False)
    reading_order = Column(Integer, nullable=False)
    polygon_json = Column(Text, default="[]")
    crop_artifact_ids_json = Column(Text, default="{}")
    is_crossed_out_candidate = Column(Integer, default=0)
    crossed_out_score = Column(Float, default=0.0)
    metadata_json = Column(Text, default="{}")
    created_at = Column(DateTime, nullable=False)


class RecognitionRunRow(Base):
    __tablename__ = "recognition_runs"
    id = Column(String, primary_key=True)
    line_id = Column(String, nullable=False, index=True)
    provider = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    image_variant = Column(String, nullable=False)
    source_crop_id = Column(String, nullable=False)
    configuration_json = Column(Text, default="{}")
    created_at = Column(DateTime, nullable=False)


class HypothesisRow(Base):
    __tablename__ = "hypotheses"
    id = Column(String, primary_key=True)
    recognition_run_id = Column(String, nullable=False, index=True)
    line_id = Column(String, nullable=False, index=True)
    text = Column(Text, nullable=False)
    normalized_text = Column(Text, nullable=False)
    rank = Column(Integer, nullable=False)
    sequence_score = Column(Float, nullable=False)
    visual_score = Column(Float, nullable=False)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    source_crop_id = Column(String, nullable=False)
    image_variant = Column(String, nullable=False)
    token_confidences_json = Column(Text, default="[]")
    configuration_json = Column(Text, default="{}")
    created_at = Column(DateTime, nullable=False)


class WordEvidenceRow(Base):
    __tablename__ = "word_evidence"
    id = Column(String, primary_key=True)
    line_id = Column(String, nullable=False, index=True)
    page_id = Column(String, nullable=False)
    document_id = Column(String, nullable=False, index=True)
    selected_hypothesis_id = Column(String, nullable=True)
    text = Column(Text, nullable=False)
    decision_state = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    uncertainty = Column(Float, nullable=False)
    final_score = Column(Float, nullable=False)
    alternatives_json = Column(Text, default="[]")
    evidence_ids_json = Column(Text, default="[]")
    crop_artifact_id = Column(String, nullable=True)
    decision_reason = Column(Text, default="")
    is_high_risk_entity = Column(Integer, default=0)
    reading_order = Column(Integer, default=0)
    bbox_json = Column(Text, nullable=True)
    region_type = Column(String, default="main_handwriting")
    score_breakdown_json = Column(Text, default="{}")
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class DecisionRow(Base):
    __tablename__ = "decisions"
    id = Column(String, primary_key=True)
    word_evidence_id = Column(String, nullable=False, index=True)
    state = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    uncertainty = Column(Float, nullable=False)
    reason = Column(Text, default="")
    thresholds_json = Column(Text, default="{}")
    score_breakdown_json = Column(Text, default="{}")
    created_at = Column(DateTime, nullable=False)


class ReviewTaskRow(Base):
    __tablename__ = "review_tasks"
    id = Column(String, primary_key=True)
    document_id = Column(String, nullable=False, index=True)
    word_evidence_id = Column(String, nullable=False)
    status = Column(String, default="open")
    priority = Column(Integer, default=0)
    created_at = Column(DateTime, nullable=False)


class CorrectionRow(Base):
    __tablename__ = "corrections"
    id = Column(String, primary_key=True)
    word_evidence_id = Column(String, nullable=False, index=True)
    previous_text = Column(Text, nullable=False)
    new_text = Column(Text, nullable=False)
    previous_state = Column(String, nullable=False)
    new_state = Column(String, nullable=False)
    actor = Column(String, default="user")
    created_at = Column(DateTime, nullable=False)


class AuditRow(Base):
    __tablename__ = "audit_events"
    id = Column(String, primary_key=True)
    document_id = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False)
    payload_json = Column(Text, default="{}")
    created_at = Column(DateTime, nullable=False)


def _j(obj: Any) -> str:
    return json.dumps(obj, default=str)


def _l(s: Optional[str]) -> Any:
    if not s:
        return None
    return json.loads(s)


class Repository:
    def __init__(self, db_url: str):
        connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        self.engine = create_engine(db_url, connect_args=connect_args, future=True)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

    def session(self) -> Session:
        return self.SessionLocal()

    # ---- Documents / Jobs ----
    def save_document(self, doc: Document) -> Document:
        with self.session() as s:
            row = DocumentRow(
                id=doc.id,
                filename=doc.filename,
                media_type=doc.media_type,
                source_artifact_id=doc.source_artifact_id,
                status=doc.status.value,
                page_count=doc.page_count,
                error_message=doc.error_message,
                created_at=doc.created_at.replace(tzinfo=None) if doc.created_at.tzinfo else doc.created_at,
                updated_at=doc.updated_at.replace(tzinfo=None) if doc.updated_at.tzinfo else doc.updated_at,
                metadata_json=_j(doc.metadata),
            )
            s.merge(row)
            s.commit()
        return doc

    def get_document(self, document_id: str) -> Optional[Document]:
        with self.session() as s:
            row = s.get(DocumentRow, document_id)
            if not row:
                return None
            return Document(
                id=row.id,
                filename=row.filename,
                media_type=row.media_type,
                source_artifact_id=row.source_artifact_id,
                status=JobStatus(row.status),
                page_count=row.page_count,
                error_message=row.error_message,
                created_at=row.created_at,
                updated_at=row.updated_at,
                metadata=_l(row.metadata_json) or {},
            )

    def update_document_status(
        self,
        document_id: str,
        status: JobStatus,
        *,
        error_message: Optional[str] = None,
        page_count: Optional[int] = None,
    ) -> None:
        with self.session() as s:
            row = s.get(DocumentRow, document_id)
            if not row:
                return
            row.status = status.value
            row.updated_at = utcnow().replace(tzinfo=None)
            if error_message is not None:
                row.error_message = error_message
            if page_count is not None:
                row.page_count = page_count
            s.commit()

    def upsert_job(
        self,
        job_id: str,
        document_id: str,
        status: str,
        stage: str,
        progress: float,
        message: str = "",
        error_message: Optional[str] = None,
    ) -> None:
        now = utcnow().replace(tzinfo=None)
        with self.session() as s:
            row = s.get(JobRow, job_id)
            if row is None:
                row = JobRow(
                    id=job_id,
                    document_id=document_id,
                    status=status,
                    stage=stage,
                    progress=progress,
                    message=message,
                    error_message=error_message,
                    created_at=now,
                    updated_at=now,
                )
                s.add(row)
            else:
                row.status = status
                row.stage = stage
                row.progress = progress
                row.message = message
                row.error_message = error_message
                row.updated_at = now
            s.commit()

    def get_job_by_document(self, document_id: str) -> Optional[dict[str, Any]]:
        with self.session() as s:
            row = s.execute(
                select(JobRow).where(JobRow.document_id == document_id).order_by(JobRow.created_at.desc())
            ).scalars().first()
            if not row:
                return None
            return {
                "id": row.id,
                "document_id": row.document_id,
                "status": row.status,
                "stage": row.stage,
                "progress": row.progress,
                "message": row.message,
                "error_message": row.error_message,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }

    # ---- Artifacts ----
    def save_artifact(self, artifact: Artifact) -> Artifact:
        with self.session() as s:
            row = ArtifactRow(
                id=artifact.id,
                sha256=artifact.sha256,
                path=artifact.path,
                media_type=artifact.media_type,
                source_artifact_id=artifact.source_artifact_id,
                transformation_name=artifact.transformation_name,
                settings_json=_j(artifact.settings),
                created_at=artifact.created_at.replace(tzinfo=None) if artifact.created_at.tzinfo else artifact.created_at,
                page_id=artifact.page_id,
                document_id=artifact.document_id,
            )
            s.merge(row)
            s.commit()
        return artifact

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        with self.session() as s:
            row = s.get(ArtifactRow, artifact_id)
            if not row:
                return None
            return Artifact(
                id=row.id,
                sha256=row.sha256,
                path=row.path,
                media_type=row.media_type,
                source_artifact_id=row.source_artifact_id,
                transformation_name=row.transformation_name,
                settings=_l(row.settings_json) or {},
                created_at=row.created_at,
                page_id=row.page_id,
                document_id=row.document_id,
            )

    # ---- Pages / Regions / Lines ----
    def save_page(self, page: Page) -> Page:
        with self.session() as s:
            row = PageRow(
                id=page.id,
                document_id=page.document_id,
                page_index=page.page_index,
                width=page.width,
                height=page.height,
                raw_artifact_id=page.raw_artifact_id,
                normalized_artifact_id=page.normalized_artifact_id,
                blur_score=page.blur_score,
                contrast_score=page.contrast_score,
                skew_angle=page.skew_angle,
                resolution=page.resolution,
                noise_score=page.noise_score,
                quality_class=page.quality_class.value if page.quality_class else None,
                artifact_ids_json=_j(page.artifact_ids),
                preprocessing_settings_json=_j(page.preprocessing_settings),
                created_at=page.created_at.replace(tzinfo=None) if page.created_at.tzinfo else page.created_at,
            )
            s.merge(row)
            s.commit()
        return page

    def list_pages(self, document_id: str) -> list[Page]:
        with self.session() as s:
            rows = s.execute(
                select(PageRow).where(PageRow.document_id == document_id).order_by(PageRow.page_index)
            ).scalars().all()
            return [self._page_from_row(r) for r in rows]

    def _page_from_row(self, row: PageRow) -> Page:
        return Page(
            id=row.id,
            document_id=row.document_id,
            page_index=row.page_index,
            width=row.width,
            height=row.height,
            raw_artifact_id=row.raw_artifact_id,
            normalized_artifact_id=row.normalized_artifact_id,
            blur_score=row.blur_score,
            contrast_score=row.contrast_score,
            skew_angle=row.skew_angle,
            resolution=row.resolution,
            noise_score=row.noise_score,
            quality_class=QualityClass(row.quality_class) if row.quality_class else None,
            artifact_ids=_l(row.artifact_ids_json) or {},
            preprocessing_settings=_l(row.preprocessing_settings_json) or {},
            created_at=row.created_at,
        )

    def save_region(self, region: Region) -> Region:
        with self.session() as s:
            row = RegionRow(
                id=region.id,
                page_id=region.page_id,
                region_type=region.region_type.value,
                bbox_json=_j(region.bbox.as_dict()),
                linked_region_id=region.linked_region_id,
                crop_artifact_id=region.crop_artifact_id,
                metadata_json=_j(region.metadata),
                created_at=region.created_at.replace(tzinfo=None) if region.created_at.tzinfo else region.created_at,
            )
            s.merge(row)
            s.commit()
        return region

    def list_regions(self, page_id: str) -> list[Region]:
        with self.session() as s:
            rows = s.execute(select(RegionRow).where(RegionRow.page_id == page_id)).scalars().all()
            out = []
            for r in rows:
                bb = _l(r.bbox_json)
                out.append(
                    Region(
                        id=r.id,
                        page_id=r.page_id,
                        region_type=RegionType(r.region_type),
                        bbox=BoundingBox(**bb),
                        linked_region_id=r.linked_region_id,
                        crop_artifact_id=r.crop_artifact_id,
                        metadata=_l(r.metadata_json) or {},
                        created_at=r.created_at,
                    )
                )
            return out

    def save_line(self, line: Line) -> Line:
        with self.session() as s:
            row = LineRow(
                id=line.id,
                page_id=line.page_id,
                region_id=line.region_id,
                bbox_json=_j(line.bbox.as_dict()),
                reading_order=line.reading_order,
                polygon_json=_j(line.polygon),
                crop_artifact_ids_json=_j(line.crop_artifact_ids),
                is_crossed_out_candidate=1 if line.is_crossed_out_candidate else 0,
                crossed_out_score=line.crossed_out_score,
                metadata_json=_j(line.metadata),
                created_at=line.created_at.replace(tzinfo=None) if line.created_at.tzinfo else line.created_at,
            )
            s.merge(row)
            s.commit()
        return line

    def list_lines(self, page_id: str) -> list[Line]:
        with self.session() as s:
            rows = s.execute(
                select(LineRow).where(LineRow.page_id == page_id).order_by(LineRow.reading_order)
            ).scalars().all()
            out = []
            for r in rows:
                bb = _l(r.bbox_json)
                out.append(
                    Line(
                        id=r.id,
                        page_id=r.page_id,
                        region_id=r.region_id,
                        bbox=BoundingBox(**bb),
                        reading_order=r.reading_order,
                        polygon=_l(r.polygon_json) or [],
                        crop_artifact_ids=_l(r.crop_artifact_ids_json) or {},
                        is_crossed_out_candidate=bool(r.is_crossed_out_candidate),
                        crossed_out_score=r.crossed_out_score,
                        metadata=_l(r.metadata_json) or {},
                        created_at=r.created_at,
                    )
                )
            return out

    def save_recognition_run(self, run: RecognitionRun) -> RecognitionRun:
        with self.session() as s:
            row = RecognitionRunRow(
                id=run.id,
                line_id=run.line_id,
                provider=run.provider,
                model_name=run.model_name,
                model_version=run.model_version,
                image_variant=run.image_variant,
                source_crop_id=run.source_crop_id,
                configuration_json=_j(run.configuration),
                created_at=run.created_at.replace(tzinfo=None) if run.created_at.tzinfo else run.created_at,
            )
            s.merge(row)
            s.commit()
        return run

    def save_hypothesis(self, hyp: Hypothesis, line_id: str) -> Hypothesis:
        with self.session() as s:
            row = HypothesisRow(
                id=hyp.id,
                recognition_run_id=hyp.recognition_run_id,
                line_id=line_id,
                text=hyp.text,
                normalized_text=hyp.normalized_text,
                rank=hyp.rank,
                sequence_score=hyp.sequence_score,
                visual_score=hyp.visual_score,
                model_name=hyp.model_name,
                model_version=hyp.model_version,
                source_crop_id=hyp.source_crop_id,
                image_variant=hyp.image_variant,
                token_confidences_json=_j(hyp.token_confidences),
                configuration_json=_j(hyp.configuration),
                created_at=hyp.created_at.replace(tzinfo=None) if hyp.created_at.tzinfo else hyp.created_at,
            )
            s.merge(row)
            s.commit()
        return hyp

    def list_hypotheses_for_line(self, line_id: str) -> list[Hypothesis]:
        with self.session() as s:
            rows = s.execute(
                select(HypothesisRow).where(HypothesisRow.line_id == line_id).order_by(HypothesisRow.rank)
            ).scalars().all()
            return [
                Hypothesis(
                    id=r.id,
                    recognition_run_id=r.recognition_run_id,
                    text=r.text,
                    normalized_text=r.normalized_text,
                    rank=r.rank,
                    sequence_score=r.sequence_score,
                    visual_score=r.visual_score,
                    model_name=r.model_name,
                    model_version=r.model_version,
                    source_crop_id=r.source_crop_id,
                    image_variant=r.image_variant,
                    token_confidences=_l(r.token_confidences_json) or [],
                    configuration=_l(r.configuration_json) or {},
                    created_at=r.created_at,
                )
                for r in rows
            ]

    def save_word_evidence(self, we: WordEvidence) -> WordEvidence:
        with self.session() as s:
            row = WordEvidenceRow(
                id=we.id,
                line_id=we.line_id,
                page_id=we.page_id,
                document_id=we.document_id,
                selected_hypothesis_id=we.selected_hypothesis_id,
                text=we.text,
                decision_state=we.decision_state.value,
                confidence=we.confidence,
                uncertainty=we.uncertainty,
                final_score=we.final_score,
                alternatives_json=_j(we.alternatives),
                evidence_ids_json=_j(we.evidence_ids),
                crop_artifact_id=we.crop_artifact_id,
                decision_reason=we.decision_reason,
                is_high_risk_entity=1 if we.is_high_risk_entity else 0,
                reading_order=we.reading_order,
                bbox_json=_j(we.bbox.as_dict()) if we.bbox else None,
                region_type=we.region_type.value,
                score_breakdown_json=_j(we.score_breakdown),
                created_at=we.created_at.replace(tzinfo=None) if we.created_at.tzinfo else we.created_at,
                updated_at=we.updated_at.replace(tzinfo=None) if we.updated_at.tzinfo else we.updated_at,
            )
            s.merge(row)
            s.commit()
        return we

    def get_word_evidence(self, word_id: str) -> Optional[WordEvidence]:
        with self.session() as s:
            r = s.get(WordEvidenceRow, word_id)
            if not r:
                return None
            return self._word_from_row(r)

    def list_word_evidence(self, document_id: str) -> list[WordEvidence]:
        with self.session() as s:
            rows = s.execute(
                select(WordEvidenceRow)
                .where(WordEvidenceRow.document_id == document_id)
                .order_by(WordEvidenceRow.reading_order)
            ).scalars().all()
            return [self._word_from_row(r) for r in rows]

    def _word_from_row(self, r: WordEvidenceRow) -> WordEvidence:
        bb = _l(r.bbox_json)
        return WordEvidence(
            id=r.id,
            line_id=r.line_id,
            page_id=r.page_id,
            document_id=r.document_id,
            selected_hypothesis_id=r.selected_hypothesis_id,
            text=r.text,
            decision_state=DecisionState(r.decision_state),
            confidence=r.confidence,
            uncertainty=r.uncertainty,
            final_score=r.final_score,
            alternatives=_l(r.alternatives_json) or [],
            evidence_ids=_l(r.evidence_ids_json) or [],
            crop_artifact_id=r.crop_artifact_id,
            decision_reason=r.decision_reason,
            is_high_risk_entity=bool(r.is_high_risk_entity),
            reading_order=r.reading_order,
            bbox=BoundingBox(**bb) if bb else None,
            region_type=RegionType(r.region_type),
            score_breakdown=_l(r.score_breakdown_json) or {},
            created_at=r.created_at,
            updated_at=r.updated_at,
        )

    def save_decision(self, d: Decision) -> Decision:
        with self.session() as s:
            row = DecisionRow(
                id=d.id,
                word_evidence_id=d.word_evidence_id,
                state=d.state.value,
                confidence=d.confidence,
                uncertainty=d.uncertainty,
                reason=d.reason,
                thresholds_json=_j(d.thresholds_used),
                score_breakdown_json=_j(d.score_breakdown),
                created_at=d.created_at.replace(tzinfo=None) if d.created_at.tzinfo else d.created_at,
            )
            s.merge(row)
            s.commit()
        return d

    def save_review_task(self, task: ReviewTask) -> ReviewTask:
        with self.session() as s:
            row = ReviewTaskRow(
                id=task.id,
                document_id=task.document_id,
                word_evidence_id=task.word_evidence_id,
                status=task.status,
                priority=task.priority,
                created_at=task.created_at.replace(tzinfo=None) if task.created_at.tzinfo else task.created_at,
            )
            s.merge(row)
            s.commit()
        return task

    def save_correction(self, c: CorrectionRecord) -> CorrectionRecord:
        with self.session() as s:
            row = CorrectionRow(
                id=c.id,
                word_evidence_id=c.word_evidence_id,
                previous_text=c.previous_text,
                new_text=c.new_text,
                previous_state=c.previous_state.value,
                new_state=c.new_state.value,
                actor=c.actor,
                created_at=c.created_at.replace(tzinfo=None) if c.created_at.tzinfo else c.created_at,
            )
            s.merge(row)
            s.commit()
        return c

    def list_corrections(self, word_evidence_id: str) -> list[CorrectionRecord]:
        with self.session() as s:
            rows = s.execute(
                select(CorrectionRow)
                .where(CorrectionRow.word_evidence_id == word_evidence_id)
                .order_by(CorrectionRow.created_at)
            ).scalars().all()
            return [
                CorrectionRecord(
                    id=r.id,
                    word_evidence_id=r.word_evidence_id,
                    previous_text=r.previous_text,
                    new_text=r.new_text,
                    previous_state=DecisionState(r.previous_state),
                    new_state=DecisionState(r.new_state),
                    actor=r.actor,
                    created_at=r.created_at,
                )
                for r in rows
            ]

    def save_audit(self, event: AuditEvent) -> AuditEvent:
        with self.session() as s:
            row = AuditRow(
                id=event.id,
                document_id=event.document_id,
                event_type=event.event_type,
                payload_json=_j(event.payload),
                created_at=event.created_at.replace(tzinfo=None) if event.created_at.tzinfo else event.created_at,
            )
            s.merge(row)
            s.commit()
        return event
